from math import *
import numpy as np

PROPORTIONAL_GAIN = 1.0
INTEGRAL_GAIN = 0.5
# DERIVATIVE_GAIN = 0.2


class PIDLineFollower:
    def __init__(self):
        self.prev_errors = [0 for _ in range(25)]

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

    def get_control(self, angle_error, x, y, theta, speed, turn_angle):
        """Returns steering angle given the angle offset in radians from center
        :param angle_error: angle in radians
        :return: steering angle in radians (positive turns right, negative turns
        left)
        """
        derivative_error = sum([self.prev_errors[i] - self.prev_errors[i-1] for i in range(1, len(self.prev_errors))])

        control = (PROPORTIONAL_GAIN * angle_error) \
                + (INTEGRAL_GAIN * np.mean(self.prev_errors))
        # + (DERIVATIVE_GAIN * derivative_error) \

        self.prev_errors = [angle_error] + self.prev_errors[:24]
        return control