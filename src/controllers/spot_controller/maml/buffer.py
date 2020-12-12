import numpy as np
import torch
from scipy.signal import lfilter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VPGBuffer():

    def __init__(self, K: int, action_dim: int, run_length: int, obs_dim: int):
        self.gamma = 0.8
        self.lam = 0.95
        self.obs_buff = torch.zeros((K, run_length, obs_dim))
        self.logp_buff = torch.zeros((K, run_length))
        self.action_buff = torch.zeros((K, run_length, action_dim))
        self.reward_buff = torch.zeros((K, run_length))
        self.rewards_togo_buff = torch.zeros((K, run_length))
        self.ptr = 0

    def _calc_rew_togo(self):
        # TODO: MAKE ASYNC! CALL AT END OF TASK_ROLLOUT
        R = np.append(self.reward_buff[self.ptr].detach(), 0)
        self.rewards_togo_buff[self.ptr] = self._disc_cumsum(R, self.gamma)[:-1]

    def _disc_cumsum(self, x, discount):
        """Calculate discounted sum of vector
        From spinup ppo """
        return torch.from_numpy(
            lfilter(b=[1], a=[1, float(-discount)], x=x[::-1],
                    axis=0)[::-1].copy()
            ).to(device)

    def save_run(self, obs, action, logp, reward, i):
        self.obs_buff[self.ptr][i] = obs
        self.logp_buff[self.ptr][i] = logp
        self.action_buff[self.ptr][i] = action
        self.reward_buff[self.ptr][i] = reward

    def finish_rollout(self):
        self._calc_rew_togo()
        self.ptr += 1

    def get(self):
        return (
            self.obs_buff, self.logp_buff, self.action_buff,
            self.rewards_togo_buff
            )


class TRPOBuffer():

    def __init__(self, K: int, action_dim: int, run_length: int, obs_dim: int):
        self.gamma = 0.99
        self.lam = 0.95

        self.obs_buff = torch.zeros((K, run_length, obs_dim))
        self.val_buff = torch.zeros((K, run_length))
        self.last_val_buff = torch.zeros(K)

        self.action_buff = torch.zeros((K, run_length, action_dim))

        self.reward_buff = torch.zeros((K, run_length))
        self.rewards_togo_buff = torch.zeros((K, run_length))

        self.logp_buff = torch.zeros((K, run_length))
        self.adv_buff = torch.zeros((K, run_length))

        self.K_ptr = 0
        self.ptr = 0

        self.K_limit = K
        self.ptr_lim = run_length

    def save_run(
        self, action: torch.Tensor, value: torch.Tensor, obs: torch.Tensor,
        logp: torch.Tensor
        ):
        """Save observation and mu for later model update"""
        # yapf: disable
        if (self.K_ptr < self.K_limit) and (self.ptr < self.ptr_lim):
            self.obs_buff[self.K_ptr][self.ptr] = obs
            self.action_buff[self.K_ptr][self.ptr] = action
            self.val_buff[self.K_ptr][self.ptr] = value
            self.logp_buff[self.K_ptr][self.ptr] = logp
            self.ptr += 1
        # yapf: enable

    def finish_model(self, store_run: bool):
        if store_run:
            self.K_ptr = 0

    def save_post_step(self, rew):
        self.reward_buff[self.K_ptr][self.ptr - 1] = rew

    def _disc_cumsum(self, x, discount):
        """Calculate discounted sum of vector
        From spinup ppo """
        return torch.from_numpy(
            lfilter(b=[1], a=[1, float(-discount)], x=x[::-1],
                    axis=0)[::-1].copy()
            ).to(device)

    def finish_rollout(self, store_run: bool, v: torch.Tensor):
        if store_run:
            self.last_val_buff[self.K_ptr] = v
            self._calc_adv(self.K_ptr)
            self.K_ptr += 1
            self.ptr = 0

    def _calc_adv(self, i):
        # TODO: MAKE ASYNC! CALL AT END OF TASK_ROLLOUT
        deltas = torch.zeros((self.ptr_lim))
        R = np.append(
            self.reward_buff[i].detach(), self.last_val_buff[i].detach()
            )
        V = np.append(self.val_buff[i].detach(), self.last_val_buff[i].detach())
        deltas = R[:-1] + self.gamma * V[1:] - V[:-1]

        self.adv_buff[i] = self._disc_cumsum(deltas, self.gamma)
        self.rewards_togo_buff[i] = self._disc_cumsum(R, self.gamma)[:-1]

    def get_rollout_info(self, i):
        ret = (
            self.action_buff[i], self.adv_buff[i], self.val_buff[i],
            self.obs_buff[i], self.rewards_togo_buff[i], self.logp_buff[i]
            )
        return ret
