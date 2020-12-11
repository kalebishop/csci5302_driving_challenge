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
    def __init__(self, align_obs=None):
        self.wheelbase = 2.875
        self.length = 4.69
        self.overh = self.length - self.wheelbase / 2
        self.width = 2.0

        r_min = 11.8 / 2
        self.ri_min = np.sqrt(r_min ** 2 - self.wheelbase**2) - self.width/2
        self.re_min = np.sqrt((self.ri_min + self.width)**2 + (self.wheelbase + self.overh) ** 2)
        self.l_min = self.overh + np.sqrt(self.re_min**2 - self.ri_min**2)

        # self.obstacles = [
        #     Obstacle([80, 147], 2, 7),
        #     Obstacle([80, 134], 2, 5)
        # ]

        self.goal1 = np.array([83, 161])
        self.goal2 = np.array([80, 153])
        self.start = np.array([87, 138])
        self.cur_state = np.array([self.start[0], self.start[1], np.pi / 2])
        self.goal_reached = False

        self.aligned = False
        self.parked = False

        self.worldinfo_basic_timestep = 10

        _, _, _, self.s1, self.t1 = self.calculate_trajectory(self.start, self.goal1)
        _, _, _, self.s2, self.t2 = self.calculate_trajectory(self.goal1, self.goal2)
        print(self.s1, self.t1, self.s2, self.t2)

    def parallel_park(self):
        if not self.aligned:
            action = self.park_control(self.s1, self.t1, self.start, self.goal1)
            if action[0] == 0 and action[1] == 0:
                self.aligned = True
                print("ALIGNED")
                self.goal_reached = False
        elif not self.parked:
            action = self.park_control(self.s2, self.t2, self.goal1, self.goal2)
            if action[0] == 0 and action[1] == 0:
                self.parked = True
                print("PARKED")
        else:
            action = (0, 0)
        return action

    def observation_tf(self, pos):
        # pos as relative [x, z] to car
        delta = pos - self.cur_state[:2]
        r = np.linalg.norm(delta)
        phi = self.cur_state[2] - np.arctan2(delta[1], delta[0])
        x, y, theta = self.cur_state
        landmark_pos = np.array([x, y]) + np.array([r * np.sin(phi), r * np.cos(phi)])
        return landmark_pos

    def calculate_trajectory(self, initial_pt, goal):
        r_prime = self.ri_min + self.width/2
        y_dir = (goal - initial_pt)[1] / abs((goal - initial_pt)[1])
        
        c1 = np.array([goal[0] + r_prime, goal[1]])

        c2_x = initial_pt[0] - r_prime
        
        t_x = (c1[0] + c2_x) / 2
        t_y = c1[1] - y_dir * np.sqrt(r_prime ** 2 - (t_x - c1[0]) ** 2)

        s_y = 2 * t_y - c1[1]
        s_x = initial_pt[0]
        c2_y = s_y

        c2 = np.array(c2_x, c2_y)
        start_pt = np.array([s_x, s_y])
        trans_pt = np.array([t_x, t_y])
        return c1, c2, initial_pt, start_pt, trans_pt

    def park_control(self, start_pt, trans_pt, initial, goal_pt):
        update_pt = self.cur_state[:2]

        if (goal_pt - initial)[1] >= 0:
            # pulling forward
            if update_pt[1] < start_pt[1]:
                # pull to start
                return (5, 0.0)

            elif update_pt[1] >= start_pt[1] and update_pt[1] < trans_pt[1]:
                # turn in left
                return (5, -0.5)

            elif update_pt[1] >= trans_pt[1] and update_pt[1] < (goal_pt[1]) \
                and not self.goal_reached:
                # turn in right
                return (5, 0.5)

            elif update_pt[1] >= goal_pt[1] and not self.goal_reached:
                # pull back - done
                self.goal_reached = True
                return (-5, 0.0)
        else:
            # pulling backward
            if update_pt[1] > start_pt[1]:
                return(-5, 0.0)

            elif update_pt[1] <= start_pt[1] and update_pt[1] > trans_pt[1]:
                # turn in left
                return (-5, -0.5)

            elif update_pt[1] <= trans_pt[1] and update_pt[1] > (goal_pt[1] - 0.5) \
                and not self.goal_reached:
                # turn in right
                return (-5, 0.5)

            elif update_pt[1] <= goal_pt[1] - 0.5 and not self.goal_reached:
                # pull back - done
                self.goal_reached = True
                return (-5, 0.0)

        if self.goal_reached and abs(self.cur_state[2] - np.pi/2) > 0.01:
            diff = self.cur_state[2] - np.pi /2
            steer = 0.5 if diff >= 0 else -0.5
            return (1, steer)
        else:
            return (0, 0.0)


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
        thetap = (us / self.wheelbase) * tan(-1 * ua)

        # print(f"from ({x:.2f}, {y:.2f}, {theta:.2f}) to ({x + xp:.2f}, {y + yp:.2f}, {theta + thetap:.2f})")

        return np.array([x + xp, y + yp, (theta + thetap) % (2 * pi)])

    def update_pos(self, action):
        self.cur_state = self.motion_model(self.cur_state, action)

if __name__ == "__main__":
    parker = AckermannParker()
    parker.calculate_trajectory()