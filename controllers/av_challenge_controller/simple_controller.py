PROPORTIONAL_GAIN = 1
INTEGRAL_GAIN = 0.05
DERIVATIVE_GAIN = 0.1


class PIDLineFollower:
    def __init__(self):
        self.prev_error = 0

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
                + (DERIVATIVE_GAIN * (angle_error - self.prev_error))
        self.prev_error = angle_error
        return control
