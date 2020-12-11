from __future__ import annotations

import copy

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import lfilter

from .model import MLP_Gaussian

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer():
    """Per TASK trajectory"""

    def __init__(self, K: int, action_dim: int, run_length: int, obs_dim: int):
        self.gamma = 0.99
        self.lam = 0.95

        self.obs_buff = torch.zeros((K, run_length, obs_dim))
        self.val_buff = torch.zeros((K, run_length))
        self.last_val_buff = torch.zeros(K)
        #self.logp_buff = torch.zeros((K, run_length, action_dim))
        #self.meta_logp_buff = torch.zeros((K, run_length, action_dim))
        self.action_buff = torch.zeros((K, run_length, action_dim))
        self.reward_buff = torch.zeros((K, run_length))
        self.rewards_togo_buff = torch.zeros((K, run_length))
        self.adv_buff = torch.zeros((K, run_length))

        self.K_ptr = 0
        self.ptr = 0

        self.K_limit = K
        self.ptr_lim = run_length

    def save_run(
        self,
        action: torch.Tensor,
        value: torch.Tensor,
        obs: torch.Tensor,
        ):
        """Save observation and mu for later model update"""
        # yapf: disable
        if (self.K_ptr < self.K_limit) and (self.ptr < self.ptr_lim):
            self.obs_buff[self.K_ptr][self.ptr] = obs
            self.action_buff[self.K_ptr][self.ptr] = action
            self.val_buff[self.K_ptr][self.ptr] = value
            self.ptr += 1
            #self.logp_buff[self.K_ptr] = logp
        # yapf: enable

    def finnish_model(self, store_run: bool):
        if store_run:
            self.K_ptr = 0
            self._calc_adv()

    def save_post_step(self, rew):
        #self.meta_logp_buff[self.task_ptr - 1][self.K_ptr - 1] = logp
        self.reward_buff[self.K_ptr][self.ptr - 1] = rew

    def _disc_cumsum(self, x, discount):
        """Calculate discounted sum of vector
        From spinup ppo """
        return torch.from_numpy(
            lfilter(b=[1], a=[1, float(-discount)], x=x[::-1],
                    axis=0)[::-1].copy()
            ).to(device)

    def finnish_run(self, store_run: bool, v: torch.Tensor):
        if store_run:
            self.last_val_buff[self.K_ptr] = v
            self.K_ptr += 1
            self.ptr = 0

    def _calc_adv(self):
        # TODO: MAKE ASYNC! CALL AT END OF TASK_ROLLOUT

        # TODO: NEED TO CAHNGE

        deltas = torch.zeros((self.ptr_lim))
        for i in range(self.K_limit):
            R = np.append(
                self.reward_buff[i].detach(), self.last_val_buff[i].detach()
                )
            V = np.append(
                self.val_buff[i].detach(), self.last_val_buff[i].detach()
                )
            deltas = R[:-1] + self.gamma * V[1:] - V[:-1]

            self.adv_buff[i] = self._disc_cumsum(deltas, self.gamma)
            self.rewards_togo_buff[i] = self._disc_cumsum(R, self.gamma)[:-1]

    def get_rollout_info(self):
        ret = (
            self.action_buff,
            self.adv_buff,
            self.val_buff,
            self.obs_buff,
            self.rewards_togo_buff,
            )
        return ret


class Learner(MLP_Gaussian):
    """Model interface, for generality"""

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_out: int,
        num_layers: int,
        motor_bounds: tuple = (-0.6, 0.5)
        ):
        super(Learner,
              self).__init__(dim_in, dim_h, dim_out, num_layers, motor_bounds)

        self.action_dim = dim_out
        self.obs_dim = dim_in
        self.critic = Critic(obs_dim=dim_in)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=0.002, betas=(0.5, 0.99)
            )

        self.kl_limit = 0.05  # TODO: CHANGE
        self.backtrack_coeff = 0.1  # TODO: CHANGE
        self.backtrack_iters = 10

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, betas=(0.9, 0.99)
            )

        self.distrib: torch.distributions.Normal
        # Reference: spinningup/spinup/algos/pytorch/vpg/core.py
        self.log_std = torch.FloatTensor(-0.5 * torch.ones(dim_out))
        self.identity_mtx: torch.Tensor = torch.eye(dim_out)

    def init_run(self, task_num, K, run_len):
        self.buff = Buffer(
            K, self.action_dim, run_len, self.obs_dim
            )  #TODO: CHANGE!!

    def get_logp(self, obs, action):
        mu: torch.Tensor = self.forward(obs)
        std: torch.Tensor = torch.exp(self.log_std)
        distrib = torch.distributions.Normal(mu, std)

        return distrib.log_prob(action).sum(axis=-1)

    def step(self, obs: torch.Tensor, use_buff: bool):
        """Parameterize Gaussian with output of MLP, sample
           action from distrib.
        """

        mu: torch.Tensor = self.forward(obs)
        std: torch.Tensor = torch.exp(self.log_std)

        v = self.critic(obs)

        # Diagonal Gaussian Policies section of:
        # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
        self.distrib = torch.distributions.Normal(mu, std)

        # Sample action
        action = self.distrib.sample()  # NOTE: derivatives do pass thru this
        print("ACTION", action)
        # Can also use .rsample() which uses parameterization trick for derivative

        # Get log prob from distrib
        log_prob = self.distrib.log_prob(action).sum(axis=-1)

        if use_buff:
            self.buff.save_run(action=action, value=v, obs=obs)

        # Clamp action to fall within motor bounds
        action = torch.clamp(action, self.motor_bounds[0], self.motor_bounds[1])

        return action, log_prob

    def critic_loss(self, rewards_togo, obs):
        loss = ((self.critic(obs) - rewards_togo)**2).mean()
        return loss

    def ppo_clip(self, adv, eps):
        """PPO Clip function, g in line 6 of spinup psuedo code:
            https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """

        if adv >= 0:
            return adv * (1 + eps)
        else:
            return adv * (1 - eps)

    def get_loss(self, action, adv, obs, model):
        clip_ratio = 0.2  # TODO: CHANGE

        logp = model.get_logp(obs, action)
        old_logp = self.get_logp(obs, action)

        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss = -(torch.min(ratio * adv,
                           clip_adv)).mean()  # Min in line 6 of spinup ppo
        approx_kl = (old_logp - logp).mean()

        return loss, approx_kl

    def update_model_ppo(self, expected_rewards, model):
        """Update meta_model with Proximal policy optimization after EVERY sample_traj step."""

        target_kl = 0.01
        train_iters = 10
        # TODO: recheck spinup code, think we have to use ALL the T*K rollouts for this
        # bit, or do it for every T?
        # Do it for every T, psuedo code is just for one task remember!

        action, adv, val, obs, rew_togo = self.buff.get_rollout_info()

        for i in range(train_iters):
            loss, approx_kl = self.get_loss(
                action=action, adv=adv, obs=obs, model=model
                )

            # Reached max kl
            if approx_kl > 1.5 * target_kl:
                break

            loss.backward()
            self.optim.step()

        # TODO: CHANGE THIS ITERS TO VALU ESPECIFIC
        for i in range(train_iters):
            for j in range(obs.shape[0]):
                self.critic_optim.zero_grad()
                loss = self.critic_loss(rew_togo[j], obs[j])
                loss.backward()
                self.critic_optim.step()

    def update_meta_params(self, update_vals):
        pass

    def set_and_eval_model(self, update_vals):
        pass

    def hessian_vector_product(self, x):
        # TODO: THis
        x.backward(retain_graph=True)
        x.backward(retain_graph=True)

    def conjugate_gradient(self, policy_grad, hvp):
        """Conjugate gradient

            References:
                Section B2 of: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        """

        x = torch.zeros_like(policy_grad)
        r = policy_grad.detach().clone()
        p = r.clone()
        delta = torch.dot(r, r)

        for _ in range(10):

            z = self.hessian_vector_product(p)
            alpha = delta / (torch.dot(p, z) + 1e-6)
            x += alpha * p
            r -= alpha * z
            rdot_new = torch.dot(r, r)
            p = r + (rdot_new / delta) * p
            delta = rdot_new

        return x

    def update_model_trpo(
        self, policy_grad: torch.Tensor, sample_avg_hessian: torch.Tensor,
        model_prime: Learner
        ) -> None:

        hvp = torch.zeros(10)  # TODO: CHANGE

        x_k = self.conjugate_gradient(policy_grad, hvp)

        alpha = torch.sqrt(
            2 * self.kl_limit / (torch.dot(x_k, hvp) + 1e-5)
            )  # TODO: CHange eps

        #####################################################
        # Backtracking Line search for update params, line 9
        #####################################################
        for i in range(self.backtrack_iters):
            update_vals = alpha * self.backtrack_coeff**i
            kl, loss_new, loss_old = self.set_and_eval_model(update_vals)

            if kl <= self.kl_limit and loss_new <= loss_old:
                # If update val satisfies constaints
                self.update_meta_params(update_vals)
                break


class Critic(torch.nn.Module):

    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
            )

    def forward(self, x):
        return self.model(x)


class MetaLearner(Learner):

    def __init__(self):
        self.meta = Learner()
        self.meta_copy = None
        self.learner_copy = None
        self.epoch = -1

    def _update_meta(self, cpy):
        diff = []
        for cp, actual in zip(cpy.parameters(), self.model.parameters()):
            diff.append(actual.data - cp.data)

        for i, param in enumerate(self.meta.parameters()):
            param.data += diff[i]

    def update_model(self, update_val):
        """Update model given log prob"""
        # TODO - update model given vals
        pass

    def copy_model(self):
        self.meta_copy = copy.deepcopy(self.meta)
        self.learner_copy = copy.deepcopy(self.model)
        self.epoch += 1

    def policy_forward(self, obs):
        action, logp = self.policy.step(obs)
        return action, logp
