from math import *
import numpy as np

PROPORTIONAL_GAIN = 1
INTEGRAL_GAIN = 0.5
DERIVATIVE_GAIN = 0.00


class PIDLineFollower:
    def __init__(self):
        self.prev_errors = [0,0,0,0,0]

        self.in_powerslide = False
        self.entrance_angle = None
        self.entrance_pos = None
        self.stage = 0

    def get_road_lines(self, lines, threshold=60):
        """Returns y coordinates of road lines. We want to return the y coordinate
        because that is the center of the road. Note that the image is rotated
        counter-clockwise by 90 degrees.
        :param lines: Array of four tuples where points (x1, y1) and (x2, y2) form
        a line. Ex. [(x1, y1, x2, y2) ... ]
        :param threshold: The pixel in x direction from which to count the line as
        the road line. The default value is estimated by eye balling.
        :return: Array of points Ex. [y1, y2 ...]
        """
        points = []
        if lines is not None:
            for coord in lines:
                if coord[0] > threshold:
                    points.append(coord[1])
                if coord[2] > threshold:
                    points.append(coord[3])
        return points

    def get_control(self, angle_error):
        """Returns steering angle given the angle offset in radians from center
        :param angle_error: angle in radians
        :return: steering angle in radians (positive turns right, negative turns
        left)
        """
        control = (PROPORTIONAL_GAIN * angle_error) \
                + (DERIVATIVE_GAIN * (angle_error - np.mean(self.prev_errors))) \
                + (INTEGRAL_GAIN * np.mean(self.prev_errors))
        self.prev_errors = [angle_error] + self.prev_errors[:4]
        return control

    def enter_powerslide(self, x, y, theta, speed, turn_angle):
        self.entrance_angle = theta
        self.midpoint_angle = (theta + pi / 2) % (2*pi)
        self.exit_angle = (theta + pi) % (2*pi)

        self.entrance_pos = x
        self.in_powerslide = True
        self.stage = 0
        assert speed > 25

    def get_powerslide_control(self, x, y, theta, speed, turn_angle):
        # Stage 1: setup -- turn hard, keep throttle consistent
        theta = theta % (2 * pi)
        if self.stage == 0:
            if theta >= self.midpoint_angle:
                self.stage += 1
            else:
                return 25, pi / 2

        # Stage 2: Drift -- max throttle and counter steer to lose traction
        if self.stage == 1:
            if theta >= self.exit_angle:
                self.stage += 1
            else:
                return 100, -pi / 2

        # Stage 3: Recover -- reduce speed to regain traction and correct angle
        if self.stage == 2:
            if speed <= 25 and theta <= self.exit_angle and x > self.entrance_pos:
                self.in_powerslide = False
            else:
                return 20, theta - self.entrance_angle