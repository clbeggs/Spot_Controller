from __future__ import annotations

import copy

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import lfilter

from .model import MLP_Gaussian

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.damping = 0.1
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

    def get_logp(self, obs, action):
        mu: torch.Tensor = self.forward(obs)
        std: torch.Tensor = torch.exp(self.log_std)
        distrib = torch.distributions.Normal(mu, std)

        return distrib.log_prob(action)

    def get_action(self, obs):
        mu: torch.Tensor = self.forward(obs)
        std: torch.Tensor = torch.exp(self.log_std)

        # Diagonal Gaussian Policies section of:
        # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
        self.distrib = torch.distributions.Normal(mu, std)

        # Sample action
        action = self.distrib.sample()  # NOTE: derivatives do pass thru this
        action = torch.clamp(action, self.motor_bounds[0], self.motor_bounds[1])
        return action

    def step(self, obs: torch.Tensor):
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
        # Can also use .rsample() which uses parameterization trick for derivative

        # Get log prob from distrib
        log_prob = self.distrib.log_prob(action).sum(axis=-1)

        # Clamp action to fall within motor bounds
        action = torch.clamp(action, self.motor_bounds[0], self.motor_bounds[1])

        return action, log_prob, v

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

    def get_loss(self, action, adv, obs, model, old_logp):
        clip_ratio = 0.2  # TODO: CHANGE

        logp = self.get_logp(obs, action)

        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss = -(torch.min(ratio * adv,
                           clip_adv)).mean()  # Min in line 6 of spinup ppo
        approx_kl = (old_logp - logp).mean()

        return loss, approx_kl

    def update_model_ppo(self, buff, model):
        """Update meta_model with Proximal policy optimization after EVERY sample_traj step."""

        target_kl = 0.01
        train_iters = 10
        # TODO: recheck spinup code, think we have to use ALL the T*K rollouts for this
        # bit, or do it for every T?
        # Do it for every T, psuedo code is just for one task remember!

        action, adv, val, obs, rew_togo, logp = buff.get_rollout_info()

        for i in range(train_iters):
            loss, approx_kl = self.get_loss(
                action=action, adv=adv, obs=obs, model=model, old_logp=logp
                )

            # Reached max kl
            if approx_kl > 1.5 * target_kl:
                break

            loss.backward(retain_graph=True)
            self.optim.step()

        for i in range(train_iters):
            for j in range(obs.shape[0]):
                self.critic_optim.zero_grad()
                loss = self.critic_loss(rew_togo[j], obs[j])
                loss.backward()
                self.critic_optim.step()

    def get_flat_params_from(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        return flat_params

    def set_flat_params_to(self, model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
                )
            prev_ind += flat_size

    def hessian_vector_product(self, obs, act, old_distrib, x):
        # Compute KL
        mu = self.model(obs)
        std: torch.Tensor = torch.exp(self.log_std)
        distrib = torch.distributions.Normal(mu, std)
        kl = torch.distributions.kl.kl_divergence(distrib, old_distrib).mean()

        g = torch.autograd.grad(kl, self.model.parameters(), create_graph=True)
        flat_grad_kl = self.flat_grads(g)

        kl_val = (flat_grad_kl * x).sum()
        grads = torch.autograd.grad(kl_val, self.model.parameters())
        hess_kl = self.flat_grads(grads)

        return hess_kl + x * self.damping

    def conjugate_gradient(self, grads, act, obs, old_pi):
        """Conjugate gradient

            References:
                Section B2 of: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        """
        # Get KL div
        x = torch.zeros_like(grads)
        r = grads.detach().clone()
        p = r.clone()
        delta = torch.dot(r, r)

        for _ in range(10):
            z = self.hessian_vector_product(obs, act, old_pi, p)

            alpha = delta / (torch.dot(p, z) + 1e-6)
            x += alpha * p
            r -= alpha * z
            rdot_new = torch.dot(r, r)
            p = r + (rdot_new / delta) * p
            delta = rdot_new
        return x

    def get_loss_trpo(self, action, adv, obs, old_logp):
        logp = self.get_logp(obs, action)
        ratio = torch.exp(logp - old_logp)
        loss = -(ratio * torch.transpose(adv.expand(12, -1), 0, 1)).mean()
        return loss

    def flat_grads(self, grads):
        return torch.cat([grad.contiguous().view(-1) for grad in grads])

    def kl_loss(self, obs, act, adv, logp_old, old_distrib):
        mu = self.model(obs)
        std: torch.Tensor = torch.exp(self.log_std)
        distrib = torch.distributions.Normal(mu, std)
        logp = distrib.log_prob(act)

        ratio = torch.exp(logp - logp_old)
        loss = -(ratio * torch.transpose(adv.expand(12, -1), 0, 1)).mean()
        kl_ls = torch.distributions.kl.kl_divergence(distrib,
                                                     old_distrib).mean()

        return loss, kl_ls

    def update_model_trpo(self, buff) -> None:
        for i in range(buff.K_limit):
            act, adv, val, obs, rew, logp = buff.get_rollout_info(i)

            with torch.no_grad():
                mu = self.model(obs)
                std: torch.Tensor = torch.exp(self.log_std)
                old_distrib = torch.distributions.Normal(mu, std)
                old_logp = old_distrib.log_prob(act)

            loss_old = self.get_loss_trpo(act, adv, obs, old_logp)

            grads = self.flat_grads(
                torch.autograd.grad(loss_old, self.model.parameters())
                )

            x_k = self.conjugate_gradient(
                grads=grads, obs=obs, act=act, old_pi=old_distrib
                )

            Hx = self.hessian_vector_product(
                obs=obs, act=act, x=x_k, old_distrib=old_distrib
                )
            alpha = torch.sqrt(2 * self.kl_limit / (torch.dot(x_k, Hx) + 1e-5))

            old_param = self.get_flat_params_from(self.model)

            #####################################################
            # Backtracking Line search for update params, line 9
            #####################################################
            for i in range(self.backtrack_iters):
                new_param = old_param - alpha * x_k * self.backtrack_coeff**i
                self.set_flat_params_to(self.model, new_param)
                loss, kl_loss = self.kl_loss(
                    obs, act, adv, old_logp, old_distrib
                    )

                if kl_loss <= self.kl_limit and loss <= loss_old:
                    # If update val satisfies constaints
                    break

                if i == self.backtrack_iters - 1:
                    print("DOES NOT SATISFY")
                    self.set_flat_params_to(self.model, old_param)

            train_iters = 10
            for i in range(train_iters):
                for j in range(obs.shape[0]):
                    self.critic_optim.zero_grad()
                    loss = self.critic_loss(rew[j], obs[j])
                    loss.backward()
                    self.critic_optim.step()


class Critic(torch.nn.Module):

    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
            )

    def forward(self, x):
        return self.model(x)
