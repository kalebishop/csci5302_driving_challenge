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

        # distance in meters in -z axis betweeen rear camera and front camera
        self.FRONT_CAMERA_OFFSET = -0.37

    def calculate_front_offset(self, pixel_distance):
        """ Estimate angle offset in radians from center given distance from center."""
        return np.arctan(pixel_distance / self.FOCAL_LENGTH)


robot = TeslaBot()
# @BRAD @AAQUIB change map name to anything else (or None) for new map
MAP_NAME = "main_course"
fSLAM = fastSLAM(map_=MAP_NAME)

road_line_detector = Detector(np.array([0, 0, 0.65]), np.array([255.0, 1.0, 1.0]))
line_follower = PIDLineFollower()
midpoint_y = robot.front_camera.getWidth() / 2.0

mapping_min_max_speed = (5, 30)
regular_min_max_speed = (12, 60)

robot.setCruisingSpeed(80)

count = 0
vehicle_data = {}
angle_error, control, curr_speed = 0, 0, 0

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    # TESTING
    tele.display_particles(fSLAM.particles)

    prev_control = control
    count += 1

    min_s, max_s = mapping_min_max_speed if fSLAM.lap_num < 1 else regular_min_max_speed

    # Read the sensors:
    front_cam_img = np.float32(robot.front_camera.getImageArray())  # returns image as 3D array
    rear_cam_img = np.float32(robot.rear_camera.getImageArray())
    # get lines from camera image
    lines = road_line_detector.get_lines(front_cam_img)
    road_line_points = line_follower.get_road_lines(lines)
    if len(road_line_points) > 0:
        # find the error from the road line
        error = sum(road_line_points) / float(len(road_line_points)) - midpoint_y
        angle_error = robot.calculate_front_offset(error)

        curr_speed = robot.getCurrentSpeed()
        curr_speed = curr_speed if not math.isnan(curr_speed) else 0
        curr_angle = robot.getSteeringAngle()
        curr_angle = curr_angle if not math.isnan(curr_angle) else 0
        x, y , theta = fSLAM.best_particle.mu

        control = 1.0 * line_follower.get_control(angle_error, x, y, theta, curr_speed, curr_angle)

    if fSLAM.lap_num > 0:
        # Update fSLAM future directions (i.e. are we turning left right or going straight in the next x steps)
        fSLAM.update_window(curr_speed)
        # turn if turn is coming up
        if not (-(math.pi / 128) < fSLAM.directions[0] < (math.pi / 128)):
            control = control * 0.6 + fSLAM.directions[0] * 0.8 + fSLAM.directions[1] * 0.4 + fSLAM.directions[2] * 0.3
        # turn in opposite direction before turn to give larger turn radius
        elif not (-(math.pi / 128) < np.mean(fSLAM.directions[2:]) < (math.pi / 128)):
            control = control * 0.8 - np.sign(np.mean(fSLAM.directions[2:])) * math.pi / 256.
        else:
            # If we predict a straight path, take your time to correct it
            control = control * 0.8
        control = 0.9 * control + 0.1 * prev_control

    control = np.clip(control, -0.6, 0.6)

    if type(control) == tuple:
        target_speed = control[0]
        control = control[1]
    else:
        target_speed = max_s * (1 - abs(control) * 5)
        # Reduce speed if turning
        target_speed *= 0.4 if fSLAM.turn_coming_up else 1
        # Increase speed slightly if we're straightening out soon
        target_speed *= 2 if fSLAM.turn_coming_up and fSLAM.straighten_out else 1
        target_speed = max(target_speed,  min_s)

    # brake and reduce speed when turning
    robot.setBrakeIntensity(max( (robot.getCurrentSpeed() - target_speed) / robot.getCurrentSpeed(), 0))

    # print(target_speed, control, max( (robot.getCurrentSpeed() - target_speed) / robot.getCurrentSpeed(), 0))

    robot.setCruisingSpeed(target_speed)
    # print("steering angle: %f" % control)
    robot.setSteeringAngle(control)

    visual_landmarks = []
    side_pieces = []
    # Get camera objects for visualSLAM
    visual_landmarks_front = robot.front_camera.getRecognitionObjects()
    visual_landmarks_rear = robot.rear_camera.getRecognitionObjects()

    for obj in visual_landmarks_front:
        # ignore road, barriers and whatever twoers are
        obj_model = obj.get_model()
        pos_on_image = obj.get_position_on_image()
        if not ((1. / 3) * robot.CAM_WIDTH < pos_on_image[0] < (2. / 3) * robot.CAM_WIDTH):
            continue
        if b'road' in obj_model or b'crash barrier' in obj_model:
            side_pieces.append((obj.get_id(), obj.get_position()))
            continue
        if b'twoer' in obj_model: # I have no idea what twoer is, but it shows up quite a bit???
            continue
        visual_landmarks.append((obj.get_id(), obj.get_position()))

    for obj in visual_landmarks_rear:
        obj_model = obj.get_model()
        pos_on_image = obj.get_position_on_image()
        if not ( (1./3) * robot.CAM_WIDTH < pos_on_image[0] < (2./3) * robot.CAM_WIDTH ):
            continue
        if b'road' in obj_model or b'crash barrier' in obj_model:
            obj_pos = obj.get_position()
            # change object position to the front camera's reference frame
            obj_pos[2] += robot.FRONT_CAMERA_OFFSET
            obj_pos[2] = -obj_pos[2]
            obj_pos[0] = -obj_pos[0]
            side_pieces.append((obj.get_id(), obj_pos))
            continue
        if b'twoer' in obj_model:
            continue
        obj_pos = obj.get_position()
        # change object position to the front camera's reference frame
        obj_pos[2] += robot.FRONT_CAMERA_OFFSET
        obj_pos[2] = -obj_pos[2]
        obj_pos[0] = -obj_pos[0]
        visual_landmarks.append((obj.get_id(), obj_pos))

    if len(visual_landmarks) < 20:
        visual_landmarks += side_pieces[:20 - len(visual_landmarks)]

    if count % 25 == 0:
        print(f"number of landmarks: {len(visual_landmarks)}")

    curr_speed = robot.getCurrentSpeed()
    curr_speed = curr_speed if not math.isnan(curr_speed) else 0
    curr_angle = robot.getSteeringAngle()
    curr_angle = curr_angle if not math.isnan(curr_angle) else 0

    action = (curr_speed, curr_angle)
    fSLAM.next_state(visual_landmarks, action)

    vehicle_data["speed"] =  str(curr_speed)
    vehicle_data["steer angle"] = str(curr_angle)

    tele.display_statistics(vehicle_data)

# Enter here exit cleanup code.
