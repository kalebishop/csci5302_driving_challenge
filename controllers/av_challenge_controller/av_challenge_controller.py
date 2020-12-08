#!/usr/bin/env python3
"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor
from tqdm import tqdm
from vehicle import Driver
import numpy as np
from debug_visual_slam import fastSLAM

import math
from image_filtering import Detector
from simple_controller import PIDLineFollower


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

    def calculate_front_offset(self, pixel_distance):
        """ Estimate angle offset in radians from center given distance from center."""
        return np.arctan(pixel_distance / self.FOCAL_LENGTH)


robot = TeslaBot()
fSLAM = fastSLAM()
# supervisor = Supervisor()

# lidar_width = lidar.getHorizontalResolution()
# lidar_max_range = lidar.getMaxRange()
road_line_detector = Detector(np.array([0, 0, 0.65]), np.array([255.0, 1.0, 1.0]))
line_follower = PIDLineFollower()

# midpoint of y dimension from camera
midpoint_y = robot.front_camera.getWidth() / 2.0
# print(robot.getCurrentPosition())


robot.setCruisingSpeed(40)

pbar = tqdm(total=10000)
count = 0

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    count += 1
    if count % 100 == 0:
        print(f"Step count: {count}")
    pbar.update(1)
    # Read the sensors:
    front_cam_img = np.float32(robot.front_camera.getImageArray())  # returns image as 3D array
    rear_cam_img = np.float32(robot.rear_camera.getImageArray())
    lidar_data = np.array(robot.lidar.getRangeImage())

    # print(robot.find_road_center(front_cam_img))
    # get lines from camera image
    lines = road_line_detector.get_lines(front_cam_img)
    road_line_points = line_follower.get_road_lines(lines)

    if len(road_line_points) > 0:
        # find the error from the road line
        error = sum(road_line_points) / float(len(road_line_points)) - midpoint_y
        angle_error = robot.calculate_front_offset(error)
        control = line_follower.get_control(angle_error)
        if abs(control) > 0.2:
            # brake and reduce speed when turning
            robot.setBrakeIntensity(0.5)
            robot.setCruisingSpeed(10)
        else:
            # remove braking
            robot.setBrakeIntensity(0)
            # set speed to 30 or lidar reading from center point - 20
            # the max reading from center point is 80
            robot.setCruisingSpeed(max(30, lidar_data[int(len(lidar_data) / 2)] - 20))
        # print("steering angle: %f" % control)
        robot.setSteeringAngle(control)

    visual_landmarks = []
    # Get camera objects for visualSLAM
    visual_landmarks_front = robot.front_camera.getRecognitionObjects()
    visual_landmarks_rear = robot.rear_camera.getRecognitionObjects()
    # print(visual_landmarks_front, visual_landmarks_back)

    for obj in visual_landmarks_front + visual_landmarks_rear:
        # ignore road, barriers and whatever twoers are
        obj_model = obj.get_model()
        if b'road' in obj_model:
            continue
        if b'crash barrier' in obj_model:
            continue
        if b'twoer' in obj_model: # I have no idea what twoer is, but it shows up quite a bit???
            continue
        # if b'building' in obj_model: # ???
        #     continue
        # id = obj.get_id()
        # pos = obj.get_position()
        # ori = obj.get_orientation()
        # size = obj.get_size()
        # pos_on_img = obj.get_position_on_image()
        # size_on_img = obj.get_size_on_image()
        # nc = obj.get_number_of_colors()
        # cs = obj.get_colors()
        # model = obj.get_model()
        #
        # print(f"id: {id}, pos: {pos}, ori: {ori}, size: {size}\n"
        #       f"pos_on_img: {pos_on_img}, size_on_img: {size_on_img}\n"
        #       f"num_colours: {nc}, colours: {cs}\n"
        #       f"model: {model}")
        visual_landmarks.append(obj)
        # if obj.get_position_on_image()[0] > 100: # 100 is arbitrary
        #     visual_landmarks.append(obj)

    if count % 25 == 0:
        print(f"number of landmarks: {len(visual_landmarks)}")
    #
    # print(len(visual_landmarks_front), len(visual_landmarks))
    # visual_landmarks = visual_landmarks_front #+ visual_landmarks_back
    # print([l.get_model() for l in visual_landmarks])

    curr_speed = robot.getCurrentSpeed()
    curr_speed = curr_speed if not math.isnan(curr_speed) else 0
    curr_angle = robot.getSteeringAngle()
    curr_angle = curr_angle if not math.isnan(curr_angle) else 0

    action = (curr_speed, curr_angle)
    fSLAM.next_state(visual_landmarks, action)

# Enter here exit cleanup code.
pbar.close()