from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt
import pickle

class Landmark:
    def __init__(self, idx, mu=None, Sigma=None):
        """
        :param mu: mean position. Expected as a size 2 np array (x, y)
        :param Sigma: covariance matrix. Expected as a 2x2 np array
        """
        self.id = idx
        self.mu = mu if mu is not None else np.zeros(2)
        self.Sigma = Sigma if Sigma is not None else np.eye(2)
       
class Particle:
    def __init__(self, mu=None, landmarks=None):
        """
        :param mu: mean position. Expected as a size 3 np array (x, y, theta)
        :param landmarks: list of landmarks
        """
        self.mu = mu if mu is not None else np.zeros(3)

        assert len(self.mu) == 3
        self.landmarks = landmarks if landmarks is not None else []
        self.weight = 0.8  # TODO redefine

    def update_weight(self, new_weight):
        self.weight = new_weight

    def add_landmark(self, idx, mu, Sigma):
        l = Landmark(idx, mu, Sigma)
        self.landmarks.append(l)


class fastSLAM:
    """
    States, x or xt in code, are represented as a tuple (x, y, theta)
    """

    def __init__(self, map_=None):
        self.N = 25  # max number of particles
        self.new_pos_sigma = 0.1  # TODO adjust
        self.new_theta_sigma = 1e-3
        self.new_landmark_weight = 0.8
        self.Xs = []
        self.Ys = []
        self.particles = [Particle()]
        self.best_particle = self.particles[0]
        self.lap_num = 0
        self.curr_index = 0
        self.angle_error = 0
        self.past_start = False
        self.mu = np.zeros(3)
        self.turn_coming_up = False
        self.error = 0
        self.directions = [0,0,0,0]
        self.turn_coming_up = False
        self.straighten_out = False
        self.ten_ago_idx = 0
        self.avg_indx_update = 0

        # Initialize measurement covariance
        # TODO tune
        self.us_sigma = 0.01
        self.ua_sigma = 0.1
        self.Q = np.array([[0.8, 0],
                           [0, np.deg2rad(5.0)]])

        self.axle_length = 2.875  # in meters; needed for motion model
        # milliseconds between updates to the world in simulation
        self.worldinfo_basic_timestep = 10

        if map_ is not None:
            try:
                with open(map_ + "_map.p", "rb") as f:
                    Xs, Ys, best_landmarks = pickle.load(f)

                self.particles[0].landmarks = best_landmarks
                self.map_Xs = deepcopy(Xs)
                self.map_Ys = deepcopy(Ys)
                self.Xs = deepcopy(Xs)
                self.Ys = deepcopy(Ys)
                self.curr_index = 0
                self.lap_num += 1
                # self.us_sigma = 0.02
                # self.ua_sigma = 0.2
                print(f"Loaded map from: {map_ + '_map.p'}!")

            except FileNotFoundError:
                pass

        self.resample_particles()


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

        # Add noise
        us += np.random.normal(loc=0, scale=self.us_sigma * abs(us)) # TODO tune variance
        # @ 1  degree, you get noise of 1.01 degrees,
        # @ 10 degrees you get noise of 2.1 degrees,
        # @ 20 degrees, you get noise of 5.1 degrees
        ua += np.random.normal(loc=0, scale= np.deg2rad(self.ua_sigma * np.rad2deg(abs(ua)) + 0.1))

        # KB: the below version is taken from this site recommended
        # by Brad for car mechanics: http://planning.cs.uiuc.edu/node658.html
        xp = us * cos(theta)
        yp = us * sin(theta)
        # self.axle length is distance btw axles
        thetap = (us / self.axle_length) * tan(ua)

        # print(f"from ({x:.2f}, {y:.2f}, {theta:.2f}) to ({x + xp:.2f}, {y + yp:.2f}, {theta + thetap:.2f})")

        return np.array([x + xp, y + yp, (theta + thetap) % (2 * pi)])

    def observation_model(self, xt, j):
        """
        return the observation function h (r, phi) for landmark j, + the jacobian H
        :param xt: robot position vector
        :param j:  Landmark object
        :return : observation vector (r, phi) + Jacobian matrix H
        """
        # j_noise = j.mu + np.array([np.random.normal(loc=0, scale=0.1), np.random.normal(loc=0, scale=0.1)])

        delta = j.mu - xt[:2]
        r_pred = np.linalg.norm(delta)
        phi_pred = np.arctan2(delta[1], delta[0]) + xt[2]
        h = (r_pred, phi_pred)

        # calculate 2x2 Jacobian H
        H = np.array([[r_pred * delta[0],  r_pred * delta[1]],
                      [- delta[1],         delta[0]]])

        assert r_pred > 0
        H = H / (r_pred ** 2)
        return h, H

    def inverse_observation_model(self, xt, z):
        """
        return estimated landmark position from observation vector z
        :param xt: robot position vector
        :param z: observation vector (r, phi)
        :return: landmark position as (x, y) array
        """
        r, phi = z
        x, y, theta = xt
        landmark_pos = np.array([x, y]) + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
        return landmark_pos

    def get_estimated_position(self, p, zt):
        est_pos = []
        for js in self.get_landmarks(p, zt):
            id_, r, phi = js

            for p_l in p.landmarks:
                if id_ == p_l.id:
                    delta = p_l.mu - p.mu[:2]
                    est_pos.append(np.array([p_l.mu[0] - r * np.cos(p.mu[2] + phi),
                                             p_l.mu[1] - r * np.sin(p.mu[2] + phi),
                                             phi - np.arctan2(delta[1], delta[0])]))

        return np.mean(est_pos, axis=0)

    def process_landmarks(self, zt):
        """
        format landmarks from camera recognition output.
        :param zt: list of CameraRecognitionObjects from recognition node
        :return:   list of (id, position) tuples for each landmark
        """
        landmarks = []
        for object in zt:
            landmarks.append((object.get_id(), object.get_position()))
        return landmarks

    def get_landmarks(self, p, zt):
        """
        Associate landmarks visible in observation zt with landmarks of particle k.
        If landmark in zt has not been seen before, return None for that one.
        :param p: Particle
        :param zt: observations
        :return: list of associated landmarks
        """
        js = []
        for v_landmark in zt:
            id_, pos = v_landmark

            current_angle = p.mu[2]

            # Camera y dimension is the vertical dimension, and therefore not used
            cam_x = pos[0]
            cam_z = -pos[2]

            r = np.linalg.norm(np.array([cam_x, cam_z]))
            phi = current_angle + np.arctan2(cam_x, cam_z)

            js.append((id_, r, phi))
        return js

    def EKF_Initialize(self, idx, xt, zt):
        """
        Initialize new landmark from observation zt
        :param idx: id of new landmark
        :param xt:  current robot position vector
        :param zt:  observation vector of landmark
        """
        # instantiate new landmark
        new_landmark = Landmark(idx)
        new_landmark.mu = self.inverse_observation_model(xt, zt)
        h, H = self.observation_model(xt, new_landmark)
        new_landmark.Sigma = np.linalg.inv(H) * self.Q * (np.linalg.inv(H)).transpose()
        return new_landmark

    def EKF_Update(self, xt, j, l):
        """
        Apply EKF update to landmark l of particle p using visual landmark j
        """
        r, phi = j
        j = np.array([r, phi])

        h, H = self.observation_model(xt, l)

        Q = np.matmul(np.matmul(H, l.Sigma), H.T) + self.Q
        # print(Q)

        # print(Q.shape, H.shape)

        K = np.matmul(np.matmul(l.Sigma, H.T), np.linalg.inv(Q))

        diff = (j - h)

        # l_prime = self.inverse_observation_model(xt, j)
        weight = (np.linalg.norm(2 * pi * Q) ** (-0.5)) * np.exp(
            -0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(Q)), diff))

        if self.lap_num == 0:
            diff *= min(1, -2 / np.log10(weight + 1e-30))

            l.mu = l.mu + np.matmul(K, diff)
            l.Sigma = np.matmul((np.identity(2) - np.matmul(K, H)), l.Sigma)

        return weight

    def resample_particles(self, best_particle=None, best_weight=None, visible_landmarks=None):
        """
        Only keep likely samples
        Add new likely and random samples
        NOTE: Should always restore the set of particles to a size of self.max_particles
        """
        if best_weight is None or best_weight == 0:
            best_weight = 1.

        # Step 1 keep top 20% of samples
        N = max(1, int(len(self.particles) * 0.2))
        weights = np.array([p.weight for p in self.particles])
        inds = np.argpartition(weights, -1 * N)
        new_particles = [self.particles[i] for i in inds[-N:]]

        if best_particle is not None:
            self.Xs.append(best_particle.mu[0])
            self.Ys.append(best_particle.mu[1])
        if len(self.Xs) % 100 == 0:
            print(f"BEST WEIGHT: {best_weight}, particles kept: {len(new_particles)} / {self.N}")

        # Step 2 fill the rest up with new samples
        required_new_particles = self.N - len(new_particles)
        for i in range(required_new_particles):
            # As per Brad's answer, particles should just be copied, noise should come from motion model
            old_p = np.random.choice(new_particles)
            new_mu = old_p.mu
            # new_mu[:2] += np.random.normal(0, self.new_pos_sigma, size=2)
            # new_mu[2] += np.random.normal(0, self.new_theta_sigma)
            new_particles.append(Particle(deepcopy(new_mu), deepcopy(old_p.landmarks)))

        assert len(new_particles) == self.N
        self.particles = new_particles

    def wrap_angle(self, angle):
        """
        Wrap angle to [-pi, pi]
        :param angle: angle in radians
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_window(self, speed):
        shortest_dist, closest_point_idx = float('inf'), None
        prev_distance = float('inf')

        lookahead_distance = int(10 * self.avg_indx_update) if self.avg_indx_update > 0 else 40

        for i in range(0, 20, 1):
            ci = (self.curr_index + i) % len(self.map_Xs)
            dist = np.linalg.norm(self.best_particle.mu[:2] - np.array([self.map_Xs[ci], self.map_Ys[ci]]))
            if dist < shortest_dist:
                shortest_dist = dist
                closest_point_idx = ci
            if dist > prev_distance:
                break
            prev_distance = dist

        # self.map_mu = np.array([self.map_Xs[closest_point_idx], self.map_Ys[closest_point_idx]])

        # update the average update every ten steps by taking an average over the last 10 steps
        if len(self.Xs) % 20 == 0:
            self.avg_indx_update = (closest_point_idx - self.ten_ago_idx) / 10.
            self.ten_ago_idx = closest_point_idx

        self.curr_index = closest_point_idx

        ci = closest_point_idx

        # lookahead_distance = 20 * self.avg_indx_update

        pi_, ni = ci, ci + int(lookahead_distance)# * 1.25)

        direction_str = []
        self.directions = []
        prev_angle = np.arctan2(self.map_Ys[ni] - self.map_Ys[pi_], self.map_Xs[ni] - self.map_Xs[pi_])

        for p in self.particles:
            p.mu[0] = self.map_Xs[ci] * 0.1 + p.mu[0] * 0.9
            p.mu[1] = self.map_Ys[ci] * 0.1 + p.mu[1] * 0.9

            # if len(self.Xs) % 100 == 0:
            #     print("angle DIFF: ", prev_angle, p.mu[2])

        prev_angle = self.wrap_angle(prev_angle)

        for i in range(int(lookahead_distance), int(lookahead_distance * 10) + 2, lookahead_distance):
            pi_, ni = (ci + i) % len(self.map_Xs), (ci + i + lookahead_distance) % len(self.map_Xs)
            current_angle = self.wrap_angle(np.arctan2(self.map_Ys[ni] - self.map_Ys[pi_], self.map_Xs[ni] - self.map_Xs[pi_]))
            angle_diff = self.wrap_angle(current_angle - prev_angle)

            angle_diff = np.clip(angle_diff, -0.15, 0.15)
            if abs(angle_diff) > pi / 3:
                print(f"curent angle: {current_angle}, prev angle: {prev_angle}")
                print((self.map_Ys[ni] - self.map_Ys[pi_], self.map_Xs[ni] - self.map_Xs[pi_]))
            if angle_diff > pi / 64:
                direction_str.append(f"right:{angle_diff:.2f}")
                self.directions.append(angle_diff)
            elif angle_diff < -pi / 64:
                direction_str.append(f"left:{angle_diff:.2f}")
                self.directions.append(angle_diff)
            else:
                direction_str.append(f"straight:{angle_diff:.2f}")
                self.directions.append(angle_diff)
            prev_angle = current_angle

        self.turn_coming_up = abs(np.mean(self.directions[:2])) > (pi / 16)
        self.straighten_out = abs(np.mean(self.directions[2:])) < (pi / 32)
        #
        if len(self.Xs) % 100 == 0:
            print(f"Current location: {self.best_particle.mu}, closest location: {self.map_Xs[ci], self.map_Ys[ci]}")
            print("UPDATE distance:", self.avg_indx_update, speed, lookahead_distance)
            print(direction_str)
            # if self.turn_coming_up:
            #     print("TURN IS COMING UP")




        # PROPORTIONAL COMPONENT
        # TARGET_OFFSET = 10
        # ci = closest_point_idx
        # ti = (closest_point_idx + TARGET_OFFSET) % len(self.map_Xs)
        # P_error = self.wrap_angle(np.arctan2(self.map_Ys[ti] - cy, self.map_Xs[ti] - cx) + best_particle.mu[2])
        #
        # # INTEGRAL COMPONENT
        # I_error = 0
        # for i in range(int(TARGET_OFFSET / 2), TARGET_OFFSET):
        #     ti = (ci + i) % len(self.map_Xs)
        #     I_error += self.wrap_angle(np.arctan2(self.map_Ys[ti] - cy, self.map_Xs[ti] - cx) + best_particle.mu[2])
        #
        # I_error /= (TARGET_OFFSET / 2)
        #
        # self.curr_index = ci
        #
        # if len(self.Xs) % 25 == 0:
        #     print(f"curr point {(cx, cy)}, target_point: {(self.map_Xs[ti], self.map_Ys[ti])} p error: {P_error}, i error: {I_error}")
        #
        # self.error = min(0.99, 0.5 * P_error + 0.5 * I_error) * 0.5
        #
        # # Check for turn in the next ~3 seconds
        # ti = (ci + 500) % len(self.map_Xs)
        # self.turn_coming_up = self.wrap_angle(np.arctan2(self.map_Ys[ti] - cy, self.map_Xs[ti] - cx) + best_particle.mu[2]) > pi / 16
        #
        # if len(self.Xs) % 5 == 0:
        #     print(f"curr point {(cx, cy)}, diffs and angle: {(self.map_Ys[ti] - cy, self.map_Xs[ti] - cx)}: {np.arctan2(self.map_Ys[ti] - cy, self.map_Xs[ti] - cx)}")
        #     if self.turn_coming_up:
        #         print("TURN IS COMING UP")

    def update_particle_and_landmarks(self, p, js):
        new_weight = []
        for j_observed in js:
            id_, r, phi = j_observed

            for p_landmark in p.landmarks:
                if id_ == p_landmark.id:
                    # landmark has been seen - make update
                    w_j = self.EKF_Update(p.mu, (r, phi), p_landmark)
                    break
            else:  # Landmark is unknown
                w_j = self.new_landmark_weight
                new_landmark = self.EKF_Initialize(id_, p.mu, (r, phi))
                p.landmarks.append(new_landmark)

            new_weight.append(w_j)



        if len(new_weight) > 0:
            new_weight = np.mean(new_weight)  # take n_th root of the product of the probabilities
            p.update_weight(new_weight)

    def next_state(self, zt, ut):
        """
        zt: observations at time t (a list of (x, y) tuples indicating the distance between the current position and the landmark)
        ut: action taken at time t (us, uphi)

        other things that are used:
        ct: data association magic wand --> self.get_landmarks in this class
        Xt: list of particles --> self.particles in this class
        g: Motion model --> self.motion_model in this class
        :return:
        """
        visible_landmarks = zt
        best_weight, best_particle = 0., None

        # For each particle
        for p in self.particles:
            # Motion model update
            p.mu = self.motion_model(p.mu, ut)

            # Associate visible landmarks with the landmarks of a particle.
            # Returns tuples of (pos, landmark)
            # where pos is an np.array(x, y) that predicts where the landmark is given the visual observation
            # and landmark is either the associated landmark in the particle or and int if it's a new landmark
            js = self.get_landmarks(p, visible_landmarks)

            self.update_particle_and_landmarks(p, js)

            if p.weight > best_weight:
                best_weight, best_particle = p.weight, p

        if best_particle is None:
            best_particle = np.random.choice(self.particles)

        self.resample_particles(best_particle, best_weight, visible_landmarks)

        if best_particle.mu[0] > 25:
            self.past_start = True


        if self.past_start and 1 < best_particle.mu[0] < 10 and -10 < best_particle.mu[1] < 10:
            print("Finished lap!")
            if self.lap_num == 0:
                traj_file = "trajectory.p"
                with open(traj_file, mode='wb') as f:
                    pickle.dump((self.Xs, self.Ys, best_particle.landmarks), f)

            self.map_Xs = deepcopy(self.Xs)
            self.map_Ys = deepcopy(self.Ys)
            self.curr_index = 0

            self.lap_num += 1
            self.past_start = False
            # self.us_sigma = 0.02
            # self.ua_sigma = 0.2
            for p in self.particles:
                correction_theta = np.random.normal(0, np.deg2rad(1))
                if abs(p.mu[2]) > abs(correction_theta):
                    p.mu[2] = correction_theta
                p.mu[:2 ] = np.random.normal(0, 1, size=2)


            fig = plt.figure()
            plt.plot(self.Xs[::10], self.Ys[::10])
            fig.savefig(f'lap_{self.lap_num}.png', dpi=fig.dpi)
            print(f"Saved plot to: lap_{self.lap_num}.png")
            plt.close(fig)


        if len(self.Xs) % 500 == 0:
            fig = plt.figure()
            plt.plot(self.Xs[::10], self.Ys[::10])
            fig.savefig(f'iter_{len(self.Xs)}.png', dpi=fig.dpi)
            print(f"Saved plot to: iter_{len(self.Xs)}.png")
            plt.close(fig)
            # plt.show()

        # if self.lap_num > 0:
        #     self.update_window(best_particle)

        self.best_particle = best_particle








