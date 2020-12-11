
"""
Webots Supervisor to solve MAML
"""

####################
# Packages
####################
from controller import Supervisor
from solver import Solver
from algos.vpg import VPG
from model import MLP_Gaussian
import numpy as np
import sys


class MAML():
    """MAML """
    def __init__(self,
                 robot_controller,
                 supervisor,
                 obs_space: int = 100,  # TODO: CHANGE
                 num_epochs: int = 1000,
                 batch_size: int = 2,
                 num_samples: int = 10,
                 run_length: int = 50,
                 motor_bounds: tuple = (-3, 3),  # TODO: CHANGE
                 step_size: tuple = (1e-2, 1e-2)
                 ) -> None:

        self.robot_controller = robot_controller
        self.supervisor = supervisor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples  # equivalent to K in paper
        self.run_length = run_length

        self.obs_space = obs_space
        self.motor_bounds = motor_bounds
        self.alpha, self.beta = step_size
        
        ####################
        # Init MAML solver
        ####################
        self.algo = VPG()
        self.model = MLP_Gaussian(dim_in=obs_space,
                                  dim_h=100,
                                  dim_out=1,
                                  num_layers=3,
                                  motor_bounds=motor_bounds
                                  )

        self.solver = Solver(algo=self.algo,
                             model=self.model,
                             robot=robot_controller,
                             supervisor=supervisor)

    def get_reward(self):
        pass
        
    def train(self):
        self.solver.train(self.num_epochs, self.run_length)
        
        # TODO: Get distribution of tasks
        tasks = []
        
        # TODO: Randomly init model and goal location
        
        # TODO: 
        done = False
        while not done:
            # TODO: get batch of trajectories
            traj_batch = np.empty((self.num_samples, self.run_length))
            for traj in traj_batch:
                self.solver.sample_trajectories(self.num_samples, self.run_length)
                pass









