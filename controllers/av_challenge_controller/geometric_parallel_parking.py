import numpy as np
from math import *

class Obstacle:
    def __init__(self, position, x_length, y_length):
        self.position = position
        self.x_length = x_length
        self.y_length = y_length

    def get_boundary_positions(self):
        half_x = self.x_length/2.0
        half_y = self.y_length/2.0
        x, y = self.position
        return [[x-half_x, y-half_y], [x-half_x, y+half_y], [x+half_x, y+half_y], [x+half_x, y-half_y]]

class AckermannParker:
    def __init__(self):
        self.wheelbase = 2.875
        self.length = 4.69
        self.overh = self.length - self.wheelbase / 2
        self.width = 2.0

        r_min = 11.8 / 2
        self.ri_min = np.sqrt(r_min ** 2 - self.wheelbase**2) - self.width/2
        self.re_min = np.sqrt((self.ri_min + self.width)**2 + (self.wheelbase + self.overh) ** 2)
        self.l_min = self.overh + np.sqrt(self.re_min**2 - self.ri_min**2)

        self.obstacles = [
            Obstacle([80, 147], 2, 7),
            Obstacle([80, 134], 2, 5)
        ]

        self.goal = np.array([80, 142])
        self.start = np.array([87, 159])
        self.cur_state = np.array([self.start[0], self.start[1], 3 * np.pi / 2])
        self.goal_reached = False

        self.worldinfo_basic_timestep = 10

    def calculate_trajectory(self):
        initial_pt = self.start
        goal = self.goal
        r_prime = self.ri_min + self.width/2
        
        c1 = np.array([goal[0] + r_prime, goal[1]])

        c2_x = initial_pt[0] - r_prime
        
        t_x = (c1[0] + c2_x) / 2
        t_y = c1[1] + np.sqrt(r_prime ** 2 - (t_x - c1[0]) ** 2)

        s_y = 2 * t_y - c1[1]
        s_x = initial_pt[0]
        c2_y = s_y

        c2 = np.array(c2_x, c2_y)
        start_pt = np.array([s_x, s_y])
        trans_pt = np.array([t_x, t_y])
        return c1, c2, initial_pt, start_pt, trans_pt

    def park_control(self, start_pt, trans_pt):
        update_pt = self.cur_state[:2]
        goal_pt = self.goal
        initial = self.start

        if update_pt[1] > start_pt[1]:
            # pull forward
            return (5, 0.0)

        elif update_pt[1] <= start_pt[1] and update_pt[1] > trans_pt[1]:
            # turn in left
            return (5, -0.5)

        elif update_pt[1] <= trans_pt[1] and update_pt[1] > (goal_pt[1] - 0.5) \
            and not self.goal_reached:
            # turn in right
            return (5, 0.5)

        elif update_pt[1] <= goal_pt[1] - 0.5:
            # pull back - done
            goal_reached = True
            return (-5, 0.0)

        elif self.goal_reached:
            # pull forward - done
            return (5, 0.0)
        else:
            raise NotImplementedError

    def motion_model(self, x, u):
        """
        Given previous state x, and action u, return motion model
        denoted as g in code
        :param x: state (x, y, theta)
        :param u: action, expects it in the for (u_speed, u_change_in_steering_angle)
        :return: g(x, u)
        """
        x, y, theta = x
        us, ua = u  # speed and angle change

        # convert speed from km/hour to meters/(WorldInfo.basicTimeStep * msec)
        us = us * (1000 / 3600000.) * self.worldinfo_basic_timestep

        # KB: the below version is taken from this site recommended
        # by Brad for car mechanics: http://planning.cs.uiuc.edu/node658.html
        xp = us * cos(theta)
        yp = us * sin(theta)
        # self.axle length is distance btw axles
        thetap = (us / self.wheelbase) * tan(ua)

        # print(f"from ({x:.2f}, {y:.2f}, {theta:.2f}) to ({x + xp:.2f}, {y + yp:.2f}, {theta + thetap:.2f})")

        return np.array([x + xp, y + yp, (theta + thetap) % (2 * pi)])

    def update_pos(self, action):
        self.cur_state = self.motion_model(self.cur_state, action)