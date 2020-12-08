from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.N = 50  # max number of particles
        self.new_pos_sigma = 0.1  # TODO adjust
        self.new_theta_sigma = 1e-3
        self.new_landmark_weight = 0.8
        self.Xs = []
        self.Ys = []
        self.particles = [Particle()]
        self.resample_particles(1.)

        # Initialize measurement covariance
        # TODO tune
        self.Q = np.array([[3, 0],
                           [0, 3]])

        self.axle_length = 3  # in meters; needed for motion model
        # milliseconds between updates to the world in simulation
        self.worldinfo_basic_timestep = 10

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

        # print(us, ua)

        # convert speed from km/hour to meters/(WorldInfo.basicTimeStep * msec)
        us = us * (1000 / 3600000.) * self.worldinfo_basic_timestep

        # us = us / 3.6

        # print(us, ua)

        # Add noise
        us += np.random.normal(loc=0, scale=0.001) # TODO tune variance
        ua += np.random.normal(loc=0, scale=(pi / 180.))

        # KB: the below version is taken from this site recommended
        # by Brad for car mechanics: http://planning.cs.uiuc.edu/node658.html
        xp = us * cos(theta)
        yp = us * sin(theta)
        # self.axle length is distance btw axles
        thetap = (us / self.axle_length) * tan(ua)

        # print(f"from ({x:.2f}, {y:.2f}, {theta:.2f}) to ({x + xp:.2f}, {y + yp:.2f}, {theta + thetap:.2f})")

        return np.array([x + xp, y + yp, theta + thetap]) 

    def observation_model(self, xt, j):
        """
        return the observation function h (r, phi) for landmark j, + the jacobian H
        :param xt: robot position vector
        :param j:  Landmark object
        :return : observation vector (r, phi) + Jacobian matrix H
        """
        delta = j.mu - xt[:2]
        r_pred = np.linalg.norm(delta)
        phi_pred = np.arctan2(delta[1], delta[0]) - xt[2]
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

        l_prime = self.inverse_observation_model(xt, j)
        diff = l_prime - l.mu

        l.mu = l.mu + np.matmul(K, diff)
        l.Sigma = np.matmul((np.identity(2) - np.matmul(K, H)), l.Sigma)

        weight = (np.linalg.norm(2 * pi * Q) ** (-0.5)) * np.exp(
            -0.5 * np.matmul(np.matmul(diff.T, np.linalg.inv(Q)), diff))

        return weight

    def resample_particles(self, best_weight):
        """
        Only keep likely samples
        Add new likely and random samples
        NOTE: Should always restore the set of particles to a size of self.max_particles
        """
        if best_weight == 0:
            best_weight = 1.

        new_particles = []
        best_particle, best_weight = None, float('-inf')

        assert len(self.particles) > 0


        # Step 1 keep good samples
        for particle in self.particles:
            # Find best particle
            if particle.weight > best_weight or best_particle is None:
                best_particle, best_weight = particle, particle.weight
            # Keep particles randomly based on their weight
            # Normalize it so that the best particle has a 100% chance of being kept
            if np.random.random() < (particle.weight / best_weight):
                new_particles.append(particle)

        if len(new_particles) == 0.:
            # Always keep best particle if it isn't already kept
            new_particles.append(best_particle)

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
        best_weight = 0.

        # For each particle
        for p in self.particles:
            new_weight = []
            # Motion model update
            p.mu = self.motion_model(p.mu, ut)

            # Associate visible landmarks with the landmarks of a particle.
            # Returns tuples of (pos, landmark)
            # where pos is an np.array(x, y) that predicts where the landmark is given the visual observation
            # and landmark is either the associated landmark in the particle or and int if it's a new landmark
            js = self.get_landmarks(p, visible_landmarks)

            for j_observed in js:
                id_, r, phi = j_observed

                for p_landmark in p.landmarks:
                    if id_ == p_landmark.id:
                        # landmark has been seen - make update
                        w_j = self.EKF_Update(p.mu, (r, phi), p_landmark)
                        break
                else: # Landmark is unknown
                    w_j = self.new_landmark_weight
                    new_landmark = self.EKF_Initialize(id_, p.mu, (r, phi))
                    p.landmarks.append(new_landmark)

                new_weight.append(w_j)

            if len(new_weight) > 0:
                new_weight = np.mean(new_weight)  # take n_th root of the product of the probabilities
                p.update_weight(new_weight)

            if p.weight > best_weight:
                best_weight = p.weight

        self.resample_particles(best_weight)

        if len(self.Xs) % 500 == 0:
            fig = plt.figure()
            plt.plot(self.Xs[::10], self.Ys[::10])
            fig.savefig(f'iter_{len(self.Xs)}.png', dpi=fig.dpi)
            print(f"Saved plot to: iter_{len(self.Xs)}.png")
            # plt.show()









