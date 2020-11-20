#!/usr/bin/env python3
"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera
from vehicle import Driver
import numpy as np

# create the Robot instance.
# See Webots Driver documentation: https://www.cyberbotics.com/doc/automobile/driver-library?tab-language=python
class TeslaBot(Driver):
    def __init__(self):
        super().__init__()

        # sensors
        self.front_camera = self.getCamera("front_camera")
        self.rear_camera  = self.getCamera("rear_camera")
        self.lidar        = self.getLidar("Sick LMS 291")

        self.front_camera.enable(30)
        self.rear_camera.enable(30)
        self.lidar.enable(30)

        self.FOCAL_LENGTH = 117.4 # for default Webots 128 * 64 frontal cam

    def calculate_front_offset(self, pixel_distance):
        """ Estimate angle offset in radians from center given pixel width."""
        return np.arctan(pixel_distance / self.FOCAL_LENGTH)

robot = TeslaBot()

# lidar_width = lidar.getHorizontalResolution()
# lidar_max_range = lidar.getMaxRange()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    # Read the sensors:
    front_cam_img = np.array(robot.front_camera.getImageArray()) # returns image as 3D array
    rear_cam_img  = np.array(robot.rear_camera.getImageArray())
    lidar_data    = np.array(robot.lidar.getRangeImage())

    robot.setCruisingSpeed(10)
    
# Enter here exit cleanup code.
