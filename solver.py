from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch

from .buffer import TRPOBuffer, VPGBuffer
from .utils import plot_gradient_norms, plot_rewards

if TYPE_CHECKING:
    from ..robot import BigBrother
    from .meta import Learner


def kl_divergence(p, q):
    """pytorch implem. of kl divergence """
    return torch.distributions.kl.kl_divergence(p, q)


class Solver():

    def __init__(
        self, model: Learner, robot: BigBrother, run_length: int = 200
        ) -> None:

        self.meta_model = model
        self.model: Learner
        self.optim: torch.optim.Adam
        self.robot = robot
        self.run_length = run_length
        self.trpo_run = False
        self.decay = 0.8
        self.sample_traj_exp_rew: list = []
        self.grads: list = []

    def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
        """
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions
        """
        var0, var1 = torch.exp(2 * log_std0), torch.exp(2 * log_std1)
        pre_sum = 0.5 * (((mu1 - mu0)**2 + var0) /
                         (var1 + 1e-5) - 1) + log_std1 - log_std0
        all_kls = torch.sum(pre_sum, axis=1)
        return torch.mean(all_kls)

    def get_grad_norms(self):
        param = 0
        n = 0
        for p in self.model.parameters():
            param += p.grad.data.mean()
            n += 1
        self.grads.append(param / n)

    def update_vpg(self):
        obs, logp, act, rew_togo = self.vpg_buff.get()
        loss = torch.sum(-logp * rew_togo)
        print("=================")
        print("=================")
        print('loss inner', loss.item())
        print("=================")
        print("=================")
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def task_rollout(self):
        """Execute single episode

            Returns:
                all_rewards: List of discounted rewards * logp of taking action at each timestep
        """

        rew = 0
        for i in range(self.run_length):
            obs: tuple = self.robot.get_obs()  # Get input to model

            model_in = torch.Tensor(np.concatenate((obs[0], obs[1], obs[2])))
            # Get action and log prob of action
            action, logp, val = self.model.step(model_in)

            # Take action, get results from Webots
            step_result, terminal = self.robot.action_rollout(action)

            reward = self.robot.get_reward(obs, terminal)
            disc_rew = reward * self.decay**i
            rew += disc_rew

            # Save meta logp for PPO
            if self.trpo_run:
                self.trpo_buff.save_run(
                    action=action, value=val, obs=model_in, logp=logp
                    )
            else:
                self.vpg_buff.save_run(
                    action=action,
                    obs=model_in,
                    logp=logp,
                    reward=disc_rew,
                    i=i
                    )

            if (terminal is True) or (step_result == -1):
                print("BROKE EARLY============================")
                break
        # TODO: CHeck if need to do waht spin up did , bootstrapping v
        # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
        self.sample_traj_exp_rew.append(rew / self.run_length)
        if step_result == -1:
            v = self.model.critic(model_in)
        else:
            v = 0
        if self.trpo_run:
            self.trpo_buff.finish_rollout(self.trpo_run, v)
        else:
            self.vpg_buff.finish_rollout()

    def prep_task(self) -> None:
        """Prep models before K rollouts for single task"""

        self.model = copy.deepcopy(self.meta_model)  # Init base model

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, betas=(0.5, 0.99)
            )

        self.vpg_buff = VPGBuffer(
            action_dim=self.model.action_dim,
            obs_dim=self.model.obs_dim,
            run_length=self.run_length,
            K=self.K
            )

        self.trpo_buff = TRPOBuffer(
            action_dim=self.model.action_dim,
            obs_dim=self.model.obs_dim,
            run_length=self.run_length,
            K=self.K
            )

    def sample_trajectories(self, K: int):
        for i in range(K):
            self.robot.prep_rollout()  # Reset env before rollout
            self.task_rollout()
        self.trpo_buff.finish_model(self.trpo_run)

    def train_maml(self, tasks: list, K: int = 5, num_epochs: int = 10) -> None:
        self.K = K
        self.num_epochs = num_epochs
        self.num_tasks = len(tasks)

        for i in range(num_epochs):
            for t in range(len(tasks)):
                self.prep_task()
                self.sample_trajectories(K)

                self.update_vpg()

                self.trpo_run = True
                self.sample_trajectories(K)
                self.trpo_run = False

                # Update meta_model after each task
                # Stability trick from How to Train your MAML
                self.meta_model.update_model_trpo(buff=self.trpo_buff)
            print(
                "[%d/%d] - %f" % (i, num_epochs, self.sample_traj_exp_rew[-1])
                )
        plot_rewards(self.sample_traj_exp_rew)
        #plot_gradient_norms(self.grads)
        torch.save(
            self.meta_model.model.state_dict(),
            "/home/epiphyte/Documents/Homework/Advanced_Robotics/Final_Project/src/controllers/spot_controller/maml/weights/meta_model"
            )
