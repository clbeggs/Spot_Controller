#!/usr/bin/env python
import time

import numpy as np
from controller import Robot, Supervisor
from scipy.spatial.transform import Rotation as transform

"""
References:
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
"""
"""Constants defined for Spot from proto file"""
NUMBER_OF_LEDS = 8
NUMBER_OF_JOINTS = 12
NUMBER_OF_SENSORS = 12
NUMBER_OF_CAMERAS = 5

MOTOR_NAMES = [
    "front left shoulder abduction motor", "front left shoulder rotation motor",
    "front left elbow motor", "front right shoulder abduction motor",
    "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor", "rear left shoulder rotation motor",
    "rear left elbow motor", "rear right shoulder abduction motor",
    "rear right shoulder rotation motor", "rear right elbow motor"
    ]

SENSOR_NAMES = [
    "front left shoulder abduction sensor",
    "front left shoulder rotation sensor", "front left elbow sensor",
    "front right shoulder abduction sensor",
    "front right shoulder rotation sensor", "front right elbow sensor",
    "rear left shoulder abduction sensor", "rear left shoulder rotation sensor",
    "rear left elbow sensor", "rear right shoulder abduction sensor",
    "rear right shoulder rotation sensor", "rear right elbow sensor"
    ]

CAMERA_NAMES = [
    "left head camera", "right head camera", "left flank camera",
    "right flank camera", "rear camera"
    ]
LED_NAMES = [
    "left top led", "left middle up led", "left middle down led",
    "left bottom led", "right top led", "right middle up led",
    "right middle down led", "right bottom led"
    ]


class Spot_Node():
    """Spot Robot interface with some utils"""

    def __init__(self, robot):
        self.robot = robot

        # Init all motors, cameras, and leds
        cameras = []
        motors = []
        sensors = []
        for i in range(NUMBER_OF_CAMERAS):
            cameras.append(self.robot.getCamera(CAMERA_NAMES[i]))
        for i in range(NUMBER_OF_JOINTS):
            motors.append(self.robot.getMotor(MOTOR_NAMES[i]))
        for i in range(NUMBER_OF_SENSORS):
            sensors.append(motors[i].getPositionSensor())

        self.cameras = tuple(cameras)
        self.motors = tuple(motors)
        self.sensors = tuple(sensors)
        self.num_motors = NUMBER_OF_JOINTS
        self.num_sensors = NUMBER_OF_SENSORS

        # Enable cameras and recognition
        self.time_step = self.robot.getBasicTimeStep()
        self.cameras[0].enable(int(2 * self.time_step))
        self.cameras[1].enable(int(2 * self.time_step))
        self.cameras[0].recognitionEnable(int(2 * self.time_step))
        self.cameras[1].recognitionEnable(int(2 * self.time_step))
        for i in range(NUMBER_OF_SENSORS):
            sensors[i].enable(int(2 * self.time_step))

    def get_motor_vals(self):
        motor_vals = np.empty(self.num_motors)
        for i in range(self.num_motors):
            motor_vals[i] = self.sensors[i].getValue()
        return motor_vals


class BigBrother(Spot_Node):
    """Supervisor class for Webots """

    def __init__(self):
        self.supervisor = Supervisor()
        super().__init__(self.supervisor)

        self.spot_node = self.supervisor.getFromDef(
            "Spot_Node"
            )  # Taken from DEF in .wbt file
        self.rot_vec = self.spot_node.getField("rotation")
        self.com_node = self.supervisor.getFromDef("CentMass")
        self.goal_pt = np.array([0, 0, -16.71])
        self.goal_rad = 2

    def check_terminal(self, pos):
        if np.linalg.norm(pos - self.goal_pt) < self.goal_rad:
            return True
        else:
            return False

    def actuate_motors(self, motor_vals):
        for i, motor in enumerate(self.motors):
            # TODO - Wait for motors to reach position as given in docs: https://cyberbotics.com/doc/reference/motor?tab-language=python#wb_motor_set_position
            motor.setPosition(float(motor_vals[i]))
        return self.robot.step(int(2 * self.time_step)), self.check_terminal(
            self.spot_node.getPosition()
            )

    def action_rollout(self, motor_vals):
        self.update_com()
        return self.actuate_motors(motor_vals)

    def update_com(self):
        com = self.spot_node.getCenterOfMass()
        trans_field = self.com_node.getField("translation")
        trans_field.setSFVec3f(com)

    def reset_run_mode(self):
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_RUN)
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_PAUSE)
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()

    def reset_pause(self):
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()

    def _reset_env(self):
        """Reset physics of Sim"""
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        # self._pause_sim()

    def prep_rollout(self):
        self._reset_env()

    def get_position(self):
        return self.spot_node.getPosition()

    def get_obs(self):
        """Get observation, will be input to model"""

        self.cameras[0].recognitionEnable(int(2 * self.time_step))
        self.cameras[1].recognitionEnable(int(2 * self.time_step))

        # Get object data
        #left_cam = np.asarray(self.cameras[0].getRecognitionObjects())
        #right_cam = np.asarray(self.cameras[1].getRecognitionObjects())

        # Get motor vals
        motor_vals = np.nan_to_num(self.get_motor_vals())

        # Get robot position and center of mass
        com = np.asarray(self.spot_node.getCenterOfMass())
        pos = np.asarray(self.spot_node.getPosition())

        return (motor_vals, com, pos)

    def get_supervisor_obs(self):

        com = self.spot_node.getCenterOfMass()

        orientation = self.spot_node.getOrientation()

        position = self.spot_node.getPosition()
        velocity = self.spot_node.getVelocity()

        num_contact_points = self.spot_node.getNumberOfContactPoints()
        contact_points = [
            self.spot_node.getContactPoint(i)
            for i in range(num_contact_points)
            ]

        return {
            'contact_points': contact_points,
            'num_contact_points': num_contact_points,
            'com': com,
            'velocity': velocity,
            'position': position,
            'orientation': orientation
            }

    def get_reward(self, prev_obs: tuple, terminal: bool) -> float:
        """Get reward for MAML"""

        orientation = np.asarray(self.rot_vec.getSFRotation())  # Euler angles
        rot_z = np.abs(orientation[2])
        alpha = np.abs(orientation[3])  # Angle of rotation

        rotation_reward = rot_z * alpha  # Penalize rotation on z

        com = self.spot_node.getCenterOfMass()
        com_height: float = com[1] - 0.35

        (prev_motor_vals, com, prev_pos) = prev_obs
        cur_pos: np.ndarray = np.asarray(self.spot_node.getPosition())

        cur_pos[1] = 0  # Remove height from position
        prev_pos[1] = 0

        # largest motor change
        delta_motor = np.max(self.get_motor_vals() - prev_motor_vals)

        # Change in position
        delta_position = np.linalg.norm((cur_pos - prev_pos), ord=2)

        prev_to_goal = np.linalg.norm((prev_pos - self.goal_pt), ord=2)
        new_to_goal = np.linalg.norm((cur_pos - self.goal_pt), ord=2)
        # Distance to goal:
        delta_goal_dist: np.float32 = prev_to_goal - new_to_goal  # Will be positive if we got closer
        delta_goal_dist *= 10

        survive_reward = 0.1
        # TODO: collision

        reward_arr = np.array([
            delta_goal_dist,
            delta_position,
            -delta_motor,
            com_height,
            survive_reward,
            -rotation_reward,
            ])

        reward = reward_arr.sum()
        #print("reward function===========")
        #np.set_printoptions(precision=3)
        #print("prev pos", prev_pos)
        #print("cur pos", cur_pos)
        #print("prv 2 goal ", prev_to_goal)
        #print("cur2 goal", new_to_goal)
        #print("Change in goal dist: ", delta_goal_dist, reward_arr[0])
        #print("change in position: ", delta_position, reward_arr[1])
        #print("largest change in motor vals", -delta_motor, reward_arr[2])
        #print("Center of mass height", com_height, reward_arr[3])
        #print("survive_reward", survive_reward, reward_arr[4])
        #print("rotation reward", rotation_reward, reward_arr[5])
        #print("TOTAL REWARD: ", reward)
        #print("")

        if terminal:
            reward += 200

        return reward
