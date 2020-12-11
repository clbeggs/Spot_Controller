from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch

from .utils import plot_rewards

if TYPE_CHECKING:
    from ..robot import BigBrother
    from .meta import Learner


def kl_divergence(p, q):
    """pytorch implem. of kl divergence """
    return torch.distributions.kl.kl_divergence(p, q)


class Solver():
    """ """

    def __init__(
        self, model: Learner, robot: BigBrother, run_length: int = 25
        ) -> None:

        self.meta_model = model
        self.model: Learner
        self.optim: torch.optim.Adam
        self.robot = robot
        self.run_length = run_length
        self.store_buffer = False

        self.sample_traj_exp_rew = []

    def backtrack_line_search(self, hessian, grads, iters=10):
        """Backtracking line search for computing self.meta update val
            Args:
            Returns:
        """
        cpy = copy.deepcopy(self.meta_model)
        backtrack_coeff = 0.8  # TODO: CHANGE?
        inv_H = torch.inverse(hessian)
        print(inv_H)
        for i in range(iters):
            pass
        return None

    def conjugate_grad(self, H, g):
        """Compute Hessian of sample avg. KL divergence
            Args:
                kl_avg: Sample avg. KL-Divergence
            Returns:
                x
            Reference:
                PseudoCode @ section B2: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        """
        return None

    def gaussian_kl(self, p, q):
        """Calc. KL Divergence of two gaussians
            Args:
                p: torch.distributions.normal.Normal
                q: torch.distributions.normal.Normal
            Returns:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        """
        # TODO: Check if this actually works

        var1 = p.scale
        var2 = q.scale
        mu1 = p.loc
        mu2 = q.loc
        kl = 0.5 * (((mu2 - mu1)**2 + var1) / (var2 + 1e-5) - 1) + torch.log(
            torch.sqrt(var2)
            ) - torch.log(torch.sqrt(var1))
        return kl

    def task_rollout(self):
        """Execute single episode

            Returns:
                all_rewards: List of discounted rewards * logp of taking action at each timestep
        """
        all_rewards = torch.zeros(self.run_length)

        for i in range(self.run_length):
            obs: tuple = self.robot.get_obs()  # Get input to model

            model_in = torch.Tensor(np.concatenate((obs[0], obs[1], obs[2])))
            # Get action and log prob of action
            action, logp = self.model.step(model_in, self.store_buffer)

            # Take action, get results from Webots
            step_result, terminal = self.robot.action_rollout(action)

            reward = self.robot.get_reward(obs, terminal)
            all_rewards[i] = reward * logp

            # Save meta logp for PPO
            if self.store_buffer:
                with torch.no_grad():  # TODO: MAKE ASYNC!!!
                    self.model.buff.save_post_step(reward)

            if (terminal is True) or (step_result == -1):
                # TODO: CHeck if need to do waht spin up did , bootstrapping v
                # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
                if step_result == -1:
                    v = self.model.critic(model_in)
                else:
                    v = 0

                self.meta_model.buff.finnish_run(self.store_buffer, v)
                return torch.cumsum(all_rewards, 0)[-1]

        v = self.model.critic(model_in)
        self.meta_model.buff.finnish_run(self.store_buffer, v)
        return torch.cumsum(all_rewards, 0)[-1]

    def prep_task(self) -> None:
        """Prep models before K rollouts for single task"""
        self.model = copy.deepcopy(self.meta_model)  # Init base model
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, betas=(0.5, 0.99)
            )

    def prep_run(self) -> None:
        """Prep task rollout, reset env"""
        self.robot.prep_run()

    def sample_trajectories(self, K: int):

        rewards_for_task = torch.zeros(K)

        for i in range(K):
            self.prep_run()  # Reset env before rollout
            cumulative_reward = self.task_rollout()
            rewards_for_task[i] = cumulative_reward
            self.sample_traj_exp_rew.append(cumulative_reward)

        self.meta_model.buff.finnish_model(self.store_buffer)
        return rewards_for_task

    def train_maml(self, tasks: list, K: int = 5, num_epochs: int = 10) -> None:
        # TODO: CHANGE THIS
        K = 5
        num_epochs = 50

        for i in range(num_epochs):
            #kl_div = torch.zeros((K, self.meta_model.action_dim))
            updated_model_expec = []

            self.meta_model.init_run(len(tasks), K, self.run_length)

            for t in range(len(tasks)):
                self.prep_task()
                expec_rewards = self.sample_trajectories(K)

                # Update self.model with VPG
                model_loss = -expec_rewards.mean()
                model_loss.backward()
                self.optim.step()

                self.store_buffer = True
                expec_rewards_updated_model = self.sample_trajectories(K)
                updated_model_expec.append(expec_rewards_updated_model)
                self.store_buffer = False

                # Update meta_model after each task
                # Stability trick from How to Train your MAML
                self.meta_model.update_model_ppo(
                    expec_rewards_updated_model, self.model
                    )
        plot_rewards(self.sample_traj_exp_rew)
        torch.save(
            self.meta_model.state_dict(),
            "/home/epiphyte/Documents/Homework/Advanced_Robotics/Final_Project/src/controllers/spot_controller/maml/weights/meta_model"
            )

        # TODO: Fix KL Div.
        # KL Div. of updated and old policy
        # kl = self.gaussian_kl(self.meta_model, self.model)
        # print("action dim:", self.meta_model.action_dim)
        # print("Per task KL div shape: ", kl_div.shape)
        # print("sive of kl_div", kl_div.shape)
        # kl_div[t] = kl

        ###################################################
        # Trust region policy optimization for meta model
        ###################################################
        #print("========================================")
        #print("kl_Div shape", kl_div.shape)
        ## TODO: REMOVE
        #self.robot.reset_pause()
        #print("Last Old model reward:SOLVER: ", expec_rewards)
        #print("Updated model rewards:SOLVER: ", updated_model_expec)

        ## Sum of all expected rewards
        #meta_grad: torch.Tensor = torch.zeros(1)
        #for exp_rew in updated_model_expec:
        #    meta_grad += exp_rew.mean()
        #meta_grad *= -1  # For maximizing
        ## grads, \hat{g}_k in line 7 of TRPO pseudo code.
        #meta_grad.backward(retain_graph=True)

        #print("KL_DIV LIST size", kl_div.shape)
        #sample_avg_kl: torch.Tensor = kl_div.mean(axis=-2)

        #self.meta_model.update_model(meta_grad, sample_avg_kl, self.model)
