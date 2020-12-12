#!/usr/bin/env python3
"""parallel_parking controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor
from tqdm import tqdm
from vehicle import Driver
import numpy as np
from parallel_parking import *
from geometric_parallel_parking import AckermannParker


USE_RRT = False


# create the Robot instance.
# See Webots Driver documentation: https://www.cyberbotics.com/doc/automobile/driver-library?tab-language=python
class TeslaBot(Driver):
    def __init__(self):
        super().__init__()

        # sensors
        self.front_camera = self.getCamera("front_camera")
        self.front_camera.enable(10)
        self.front_camera.recognitionEnable(10)
        self.rear_camera = self.getCamera("rear_camera")
        self.lidar = self.getLidar("Sick LMS 291")

        self.rear_camera.enable(10)
        self.rear_camera.recognitionEnable(10)
        self.lidar.enable(10)

        self.FOCAL_LENGTH = 117.4  # for default Webots 128 * 64 frontal cam
        self.CAM_WIDTH = 128

        # distance in meters in -z axis betweeen rear camera and front camera
        self.FRONT_CAMERA_OFFSET = -0.37

    def calculate_front_offset(self, pixel_distance):
        """ Estimate angle offset in radians from center given distance from center."""
        return np.arctan(pixel_distance / self.FOCAL_LENGTH)


robot = TeslaBot()

if USE_RRT:
    rrt = RRT()
    node_list = rrt.generate_graph()
    actions = rrt.get_actions(node_list)
    print(actions)
    steps = 100
else:
    # park_controller = AckermannParker()
    align_obj = 'BMW'
    align_obj_found = False

count = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    count += 1
    if count % 100 == 0:
        print(f"Step count: {count}")

    if USE_RRT:
        if count < len(actions) * steps:
            speed, steering = actions[int(count//steps)]
            print(speed, steering)
        else:
            speed = 0
            steering = 0
        robot.setCruisingSpeed(-speed)
        robot.setSteeringAngle(steering)
    else:
        if not align_obj_found:
            objs = robot.front_camera.getRecognitionObjects()
            for o in objs:
                model = o.get_model()
                if b'BMW' in model:
                    pos = np.array([o.get_position()[0], -1 * o.get_position()[2]])
                    park_controller = AckermannParker(pos)
                    align_obj_found = True
        else:
            s, phi = park_controller.parallel_park()
            if count % 100 == 0:
                print(s, phi)
                print(park_controller.cur_state)
            robot.setCruisingSpeed(s)
            robot.setSteeringAngle(phi)
            park_controller.update_pos((s, phi))
