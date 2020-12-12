#!/usr/bin/env python

import os
import sys

import numpy as np
import torch
from controller import Supervisor

import maml
from robot import BigBrother

sys.path.append(os.path.abspath('maml'))


def rollout(model, robot, run_length):
    for i in range(run_length):
        obs: tuple = robot.get_obs()  # Get input to model

        model_in = torch.Tensor(np.concatenate((obs[0], obs[1], obs[2])))
        # Get action and log prob of action
        action, logp, val = model.step(model_in)

        # Take action, get results from Webots
        step_result, terminal = robot.action_rollout(action)


def main():

    robot = BigBrother()
    motor_bounds = (
        robot.motors[0].getMinPosition(), robot.motors[0].getMaxPosition()
        )  # -0.6, 0.5

    model = maml.Learner(
        dim_in=18,
        dim_h=256,
        dim_out=len(robot.motors),
        num_layers=1,
        motor_bounds=motor_bounds
        )
    solver = maml.Solver(model=model, robot=robot, run_length=100)
    solver.train_maml([1], K=5, num_epochs=100)

    robot.reset_run_mode()
    rollout(model, robot, 200)

    while True:
        robot.reset_run_mode()
        rollout(solver.meta_model, robot, 200)


if __name__ == "__main__":
    main()
