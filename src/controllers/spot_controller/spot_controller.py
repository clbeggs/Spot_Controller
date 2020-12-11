#!/usr/bin/env python

import os
import sys

import numpy as np
from controller import Supervisor

import maml
from robot import BigBrother

sys.path.append(os.path.abspath('maml'))


def main():
    robot = BigBrother()

    motor_bounds = (
        robot.motors[0].getMinPosition(), robot.motors[0].getMaxPosition()
        )


    model = maml.Learner(
        dim_in=18,
        dim_h=64,
        dim_out=len(robot.motors),
        num_layers=2,
        motor_bounds=motor_bounds
        )

    solver = maml.Solver(model=model, robot=robot)
    solver.train_maml([1])


if __name__ == "__main__":
    # TODO: argparse arguments
    print("SPOT CONTROLLER!!")

    main()
