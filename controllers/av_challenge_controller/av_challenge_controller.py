#!/usr/bin/env python3
"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera

# create the Robot instance.
class TeslaBot(Robot):
    def __init__(self):
        super().__init__()

        # sensors
        self.front_camera = self.getCamera("front_camera")
        self.rear_camera  = self.getCamera("rear_camera")
        self.lidar        = self.getLidar("Sick LMS 291")

        self.front_camera.enable(30)
        self.rear_camera.enable(30)
        self.lidar.enable(30)

        self.l_motor_pos  = self.getPositionSensor("left_rear_sensor")
        self.r_motor_pos  = self.getPositionSensor("right_rear_sensor")

        self.l_steer_pos  = self.getPositionSensor("left_steer_sensor")
        self.r_steer_pos  = self.getPositionSensor("right_steer_sensor")

        # motors
        self.l_motor = self.getMotor("left_rear_wheel")
        self.r_motor = self.getMotor("right_rear_wheel")

        # brakes
        self.l_brake = self.getBrake("left_rear_brake")
        self.r_brake = self.getBrake("right_rear_brake")

        # steering
        self.l_steer = self.getMotor("left_steer")
        self.r_steer = self.getMotor("right_steer")


# get the time step of the current world.
robot = TeslaBot()
timestep = int(robot.getBasicTimeStep())

# lidar_width = lidar.getHorizontalResolution()
# lidar_max_range = lidar.getMaxRange()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    pass

# Enter here exit cleanup code.
