import numpy as np
import torch

"""References:
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""

class MLP_Gaussian(torch.nn.Module):
    """Model for minimal env. just flat world, thus fixed input dim"""
    def __init__(self,
                 dim_in: int,
                 dim_h: int,
                 dim_out: int,
                 num_layers: int,
                 motor_bounds: tuple
                 ):
        super(MLP_Gaussian, self).__init__()

        ###################
        # Init Model
        ###################
        self.model = torch.nn.Sequential()
        self.model.add_module("input_layer", torch.nn.Linear(dim_in, dim_h))
        self.model.add_module("input_nonlinearity", torch.nn.ReLU())

        for i in range(num_layers):
            self.model.add_module("layer_%d" % i, torch.nn.Linear(dim_h, dim_h))
            self.model.add_module("non_linearity_%d" % i, torch.nn.ReLU())

        self.model.add_module("output_layer", torch.nn.Linear(dim_h, dim_out))
        self.log_std = -0.5 * torch.ones((dim_out))
        self.motor_bounds = motor_bounds

    def forward(self, obs):
        pred = self.model(obs)
        return pred


