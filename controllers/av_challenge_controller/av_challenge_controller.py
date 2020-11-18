#!/usr/bin/env python3
"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Display
import numpy as np

# create the Robot instance.
robot = Robot()

# sensor setup
front_camera = robot.getCamera("front_camera")
rear_camera = robot.getCamera("rear_camera")
lidar = robot.getLidar("Sick LMS 291")

# actuators (motors & brakes) & corresponding position sensors
l_motor = robot.getMotor("left_rear_wheel")
r_motor = robot.getMotor("right_rear_wheel")
l_motor_pos = robot.getPositionSensor("left_rear_sensor")
r_motor_pos = robot.getPositionSensor("right_rear_sensor")
l_brake = robot.getBrake("left_rear_brake")
r_brake = robot.getBrake("right_rear_brake")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

front_camera.enable(30)
rear_camera.enable(30)
lidar.enable(30)

lidar_width = lidar.getHorizontalResolution()
lidar_max_range = lidar.getMaxRange()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
