#!/usr/bin/env python3
"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Supervisor
from tqdm import tqdm
from vehicle import Driver
import numpy as np
from visual_SLAM import fastSLAM
from visualize_telemetry import AVTelemetry
from parallel_parking import *

import math
from image_filtering import Detector
from simple_controller import PIDLineFollower

PARALLEL_PARKING = False

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
fSLAM = fastSLAM()
tele = AVTelemetry(robot)
# supervisor = Supervisor()
robot_step = 10
steps_per_second = 1000 / robot_step

# lidar_width = lidar.getHorizontalResolution()
# lidar_max_range = lidar.getMaxRange()
road_line_detector = Detector(np.array([0, 0, 0.65]), np.array([255.0, 1.0, 1.0]))
line_follower = PIDLineFollower()

# midpoint of y dimension from camera
midpoint_y = robot.front_camera.getWidth() / 2.0
# print(robot.getCurrentPosition())

mapping_min_max_speed = (6, 30)
regular_min_max_speed = (10, 50)

# robot.setCruisingSpeed(40)

count = 0
angle_error = 0


vehicle_data = {}
# get actions by calling
rrt = RRT()
node_list = rrt.generate_graph()
actions = rrt.get_actions(node_list)
print(actions)
steps = 5 / rrt.time_step
# actions = []
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    # TESTING
    # tele.display_particles(fSLAM.particles)
    # # #
    count += 1
    if count % 100 == 0:
        print(f"Step count: {count}")

    # min_s, max_s = mapping_min_max_speed if fSLAM.lap_num < 1 else regular_min_max_speed

    # # Read the sensors:
    # front_cam_img = np.float32(robot.front_camera.getImageArray())  # returns image as 3D array
    # rear_cam_img = np.float32(robot.rear_camera.getImageArray())
    # # get lines from camera image
    # lines = road_line_detector.get_lines(front_cam_img)
    # road_line_points = line_follower.get_road_lines(lines)
    # if len(road_line_points) > 0:
    #     # find the error from the road line
    #     error = sum(road_line_points) / float(len(road_line_points)) - midpoint_y
    #     angle_error = robot.calculate_front_offset(error)
    #     control = line_follower.get_control(angle_error)

    # target_speed = max(max_s * (1 - abs(control) * 5),  min_s)
    # # if abs(control) > 0.2:
    #     # brake and reduce speed when turning
    # robot.setBrakeIntensity(max( (robot.getCurrentSpeed() - target_speed) / robot.getCurrentSpeed(), 0))

    # # print(target_speed, control, max( (robot.getCurrentSpeed() - target_speed) / robot.getCurrentSpeed(), 0))

    # robot.setCruisingSpeed(target_speed)
    # # print("steering angle: %f" % control)
    # robot.setSteeringAngle(control)


    # # parallel parking
    # # each action from rrt is run for 5 seconds = 5*step/s time steps
    if count < len(actions) * steps:
        speed, steering = actions[int(count//steps)]
        print(speed, steering)
    else:
        speed = 0
        steering = 0

    robot.setCruisingSpeed(-speed)

    robot.setSteeringAngle(steering)

    # visual_landmarks = []
    # side_pieces = []
    # # Get camera objects for visualSLAM
    # visual_landmarks_front = robot.front_camera.getRecognitionObjects()
    # visual_landmarks_rear = robot.rear_camera.getRecognitionObjects()

    # for obj in visual_landmarks_front:
    #     # ignore road, barriers and whatever twoers are
    #     obj_model = obj.get_model()
    #     pos_on_image = obj.get_position_on_image()
    #     if not ((1. / 3) * robot.CAM_WIDTH < pos_on_image[0] < (2. / 3) * robot.CAM_WIDTH):
    #         continue
    #     if b'road' in obj_model or b'crash barrier' in obj_model:
    #         side_pieces.append((obj.get_id(), obj.get_position()))
    #         continue
    #     if b'twoer' in obj_model: # I have no idea what twoer is, but it shows up quite a bit???
    #         continue
    #     # if b'building' in obj_model: # ???
    #     #     continue
    #     # id = obj.get_id()
    #     # pos = obj.get_position()
    #     # ori = obj.get_orientation()
    #     # size = obj.get_size()
    #     # pos_on_img = obj.get_position_on_image()
    #     # size_on_img = obj.get_size_on_image()
    #     # nc = obj.get_number_of_colors()
    #     # cs = obj.get_colors()
    #     # model = obj.get_model()
    #     # #
    #     # print(f"id: {id}, pos: {pos}, ori: {ori}, size: {size}\n"
    #     #       f"pos_on_img: {pos_on_img}, size_on_img: {size_on_img}\n"
    #     #       f"num_colours: {nc}, colours: {cs}\n"
    #     #       f"model: {model}")
    #     visual_landmarks.append((obj.get_id(), obj.get_position()))

    # for obj in visual_landmarks_rear:
    #     obj_model = obj.get_model()
    #     pos_on_image = obj.get_position_on_image()
    #     if not ( (1./3) * robot.CAM_WIDTH < pos_on_image[0] < (2./3) * robot.CAM_WIDTH ):
    #         continue
    #     if b'road' in obj_model or b'crash barrier' in obj_model:
    #         obj_pos = obj.get_position()
    #         # change object position to the front camera's reference frame
    #         obj_pos[2] += robot.FRONT_CAMERA_OFFSET
    #         obj_pos[2] = -obj_pos[2]
    #         obj_pos[0] = -obj_pos[0]
    #         side_pieces.append((obj.get_id(), obj_pos))
    #         continue
    #     if b'twoer' in obj_model:
    #         continue
    #     obj_pos = obj.get_position()
    #     # change object position to the front camera's reference frame
    #     obj_pos[2] += robot.FRONT_CAMERA_OFFSET
    #     obj_pos[2] = -obj_pos[2]
    #     obj_pos[0] = -obj_pos[0]
    #     visual_landmarks.append((obj.get_id(), obj_pos))

    # if len(visual_landmarks) < 20:
    #     visual_landmarks += side_pieces[:20 - len(visual_landmarks)]

    # if count % 25 == 0:
    #     print(f"number of landmarks: {len(visual_landmarks)}")


    curr_speed = robot.getCurrentSpeed()
    curr_speed = curr_speed if not math.isnan(curr_speed) else 0
    curr_angle = robot.getSteeringAngle()
    curr_angle = curr_angle if not math.isnan(curr_angle) else 0

    action = (curr_speed, curr_angle)
    # fSLAM.next_state(visual_landmarks, action)

    vehicle_data["speed"] =  str(curr_speed)
    vehicle_data["steer angle"] = str(curr_angle)

    tele.display_statistics(vehicle_data)

# Enter here exit cleanup code.
