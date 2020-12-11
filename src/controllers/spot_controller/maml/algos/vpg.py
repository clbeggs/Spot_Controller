import numpy as np

""" 
Vanilla Policy Gradient Implementation.

References:
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    https://www.janisklaise.com/post/rl-policy-gradients/
"""


class VPG():
    def __init__(self, model):
        self.model = model

    def update(self):
        pass

    def compute_loss(self, obs):
        pass

    def update(self, log_probs, obs_size):
        pass

    def train_iter(self, model):
        pass

    def model_update(self, model):
        """Compute update for model"""
        pass

    def get_reward(self, prev_pos, prev_motor, cur_pos, new_motor):
        """Reward motor change closer to goal,
        penalize large motor change values,
        penalize collisions"""

        # largest motor change
        delta_motor = np.max(new_motor - prev_motor)
        # Dist to goal
        dist = np.linalg.norm((prev_pos - cur_pos), ord=2)
        # TODO: collision


        return dist / delta_motor
