from copy import deepcopy
import numpy as np
from math import *
import matplotlib.pyplot as plt


class Landmark:
    def __init__(self, id, mu=None, Sigma=None):
        """
        :param mu: mean position. Expected as a size 2 np array (x, y)
        :param Sigma: covariance matrix. Expected as a 2x2 np array
        """
        self.id = id
        self.mu = mu if mu is not None else np.zeros(2)
        if Sigma is not None:
            self.Sigma = Sigma
        elif mu is None:
            self.Sigma = np.full((2, 2), float('inf'))
        else:
            self.Sigma = np.full((2, 2), 10.)


class Particle:
    def __init__(self, mu=None, landmarks=None):
        """
        :param mu: mean position. Expected as a size 3 np array (x, y, theta)
        :param landmarks: list of landmarks
        """
        self.mu = mu if mu is not None else np.zeros(3)
        # print("!!!", self.mu)
        assert len(self.mu) == 3
        self.landmarks = landmarks if landmarks is not None else []
        self.weight = 0.9  # TODO redefine

    def update_weight(self, new_weight):
        self.weight = new_weight

    def add_landmark(self, id, mu, Sigma):
        l = Landmark(id, mu, Sigma)
        self.landmarks.append(l)


class fastSLAM:
    """
    States, x or xt in code, are represented as a tuple (x, y, theta)
    """

    def __init__(self):
        self.N = 10  # max number of particles
        self.new_pos_sigma = 0.1  # TODO adjust
        self.new_theta_sigma = 1e-3
        self.new_landmark_weight = 0.9
        self.Xs = []
        self.Ys = []
        self.particles = [Particle()]
        self.resample_particles()

        # Initialize measurement covariance
        # TODO tune
        self.Q = np.array([[0.8, 0],
                           [0, 0.8]])

        self.axle_length = 4.792  # in meters; needed for motion model
        # milliseconds between updates to the world in simulation
        self.worldinfo_basic_timestep = 10

    def motion_model(self, x, u):
        """
        Given previous state x, and action u, return motion model
        denoted as g in code, and G is it's jacobian (first derivative with respect to state)
        :param x: state (x, y, theta)
        :param u: action, expects it in the for (u_speed, u_change_in_steering_angle)
        :return: g(x, u) and G
        """
        x, y, theta = x
        us, ua = u  # speed and angle change

        # convert speed from km/hour to meters/(WorldInfo.basicTimeStep * msec)
        us = us * 1000 / 3600000. * self.worldinfo_basic_timestep

        # Add noise
        # us += np.random.normal(loc=0, scale=0.1) # TODO tune variance
        # ua += np.random.normal(loc=0, scale=(pi / 32.))

        # KB: the below version is taken from this site recommended
        # by Brad for car mechanics: http://planning.cs.uiuc.edu/node658.html
        xp = us * cos(theta)
        yp = us * sin(theta)
        # self.axle length is distance btw axles
        thetap = us / self.axle_length * tan(ua)

        # G = np.eye(3)
        # G[0][2] = us * (-cos(theta) + cos(theta + ua))
        # G[0][1] = us * (-sin(theta) + sin(theta + ua))

        return np.array([x + xp, y + yp, theta + thetap])  # , G

    def process_landmarks(self, zt):
        """
        denoted as h in code, and H is it's jacobian (first derivative with respect to state)
        :param x: state (x, y, theta)
        :param u: action (TODO)
        :return: h(x) and H
        """
        # TODO remove cheat codes later
        landmarks = []
        for object in zt:
            landmarks.append((object.get_id(), object.get_position()))
        return landmarks

        # TODO test
        # h = [] # predicted measurement observations
        # H = [] # jacobians of observation function

        # for j in zt:
        #   delta = j.mu - x[:2]
        #   r_pred = np.linalg.norm(delta)
        #   phi_pred = np.arctan(2 * delta[1] / delta[0]) - x[2]
        #   h.append((r_pred, phi_pred))

        #   # calculate Jacobian H
        #   H.append(np.array([[-1 * r_pred * delta[0], -1 * r_pred * delta[1], 0,           r_pred * delta[0], r_pred * delta[1]],
        #                     [delta[1],               -1 * delta[0]           -1 * r_pred, -1 * delta[1],     delta[0]]]))
        #   # TODO: H might need to be mapped into a higher dim space

        # return h, H

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
            cam_z = pos[2]

            # line up the camera z axis (distance away from car) with the global x axis
            global_x = cos(current_angle) * cam_z + sin(current_angle) * cam_x
            # line up the camera x axis (distance away from car) with the global y axis
            global_y = sin(current_angle) * cam_z + cos(current_angle) * cam_x

            pos = np.array(pos[:2]) + np.array([global_x, global_y])

            for p_landmark in p.landmarks:
                if id_ == p_landmark.id:
                    js.append((pos, p_landmark))
                    # print(f"x,y prediction diff: ({pos[0] - p_landmark.mu[0]:.3f}, {pos[0] - p_landmark.mu[0]:.3f})") #landmark relative pos: {pos}, current particle pos: {p.mu}")
                    break
            else:
                js.append((pos, id_))

        return js

    def EKF_Update(self, j, l):
        """
        Apply EKF update to landmark l of particle p using visual landmark j

        In final version:
        :param h_i: predicted observation for landmark j (from self.motion_model)
        :param H_i: Jacobian for landmark j (from self.motion_model)
        :param z:   Observation vector for landmark j (vertical vector of range r, angle phi)
        """

        H = np.ones((2, 2))

        Q = np.matmul(np.matmul(H, l.Sigma), H.transpose()) + self.Q
        # print(Q)
        K = np.matmul(np.matmul(l.Sigma, H),
                      np.linalg.inv(Q))  # Kalman gain # KB: trying to use regular inverse again after fixing Q
        l.mu = l.mu + np.matmul(K, (j - l.mu))
        l.Sigma = np.matmul((np.identity(2) - np.matmul(K, H)), l.Sigma)

        # KB - alt version for when we're ready to test without location cheat -
        # TODO add appropriate results h & H from self.motion_model to params,
        # with corresponding observation z
        # l.mu = l.mu + np.matmul(K, (z - h_i))
        # l.Sigma = np.matmul((np.identity(2) - np.matmul(K, H_i)), l.Sigma)

        weight = np.linalg.norm(Q) ** (-0.5) * np.exp(
            -0.5 * np.matmul(np.matmul((j - l.mu).T, np.linalg.inv(Q)), (j - l.mu)))

        return weight

    def resample_particles(self):
        """
        Only keep likely samples
        Add new likely and random samples
        NOTE: Should always restore the set of particles to a size of self.max_particles
        """
        new_particles = []
        best_particle, best_weight = None, float('-inf')

        assert len(self.particles) > 0

        # Step 1 keep good samples
        for particle in self.particles:
            # Find best particle
            if particle.weight > best_weight:
                best_particle, best_weight = particle, particle.weight
            # Keep particles randomly based on their weight
            if np.random.random() < particle.weight:
                new_particles.append(particle)

        if len(new_particles) < self.N:
            # Always keep best particle if it isn't already kept
            new_particles.append(best_particle)
            self.Xs.append(best_particle.mu[0])
            self.Ys.append(best_particle.mu[1])

        # Step 2 fill the rest up with new samples
        required_new_particles = self.N - len(new_particles)
        for i in range(required_new_particles):
            # As per Brad's answer, particles should just be copied, noise should come from motion model
            old_p = np.random.choice(new_particles)
            new_mu = old_p.mu
            new_mu[:2] += np.random.normal(0, self.new_pos_sigma, size=2)
            new_mu[2] += np.random.normal(0, self.new_theta_sigma)
            new_particles.append(Particle(deepcopy(new_mu), deepcopy(old_p.landmarks)))

        assert len(new_particles) == self.N
        self.particles = new_particles

    def next_state(self, zt, ut):
        """
        zt: observations at time t
        ut: action taken at time t

        other things that are used:
        ct: data association magic wand --> self.get_landmarks in this class
        Xt: list of particles --> self.particles in this class
        g: Motion model --> self.motion_model in this class
        :return:
        """
        # Assumes visible landmarks is a list of landmarks
        # Specifically, a list of (x, y) tuples indicating the distance between the current position and the landmark
        visible_landmarks = self.process_landmarks(zt)

        # For each particle
        for p in self.particles:
            # Motion model update
            p.mu = self.motion_model(p.mu, ut)

            # Associate visible landmarks with the landmarks of a particle.
            # Returns tuples of (pos, landmark)
            # where pos is an np.array(x, y) that predicts where the landmark is given the visual observation
            # and landmark is either the associated landmark in the particle or and int if it's a new landmark
            js = self.get_landmarks(p, visible_landmarks)

            w_k = []

            for j in js:
                if isinstance(j[1], int):
                    new_landmark_mu = j[0]
                    new_landmark_Sigma = np.full((2, 2),
                                                 5)  # TODO redefine 5 (which is just arbitrary covariance matrix value for new landmarks)
                    p.add_landmark(j[1], new_landmark_mu, new_landmark_Sigma)
                    w_k_j = self.new_landmark_weight
                else:
                    w_k_j = self.EKF_Update(j[0], j[1])
                w_k.append(w_k_j)

            if len(w_k) > 0:
                w_k = np.power(np.prod(w_k), 1. / len(w_k))  # take n_th root of the product of the probabilities
                p.update_weight(w_k)

        self.resample_particles()

        if len(self.Xs) % 500 == 0:
            fig = plt.figure()
            plt.plot(self.Xs[::10], self.Ys[::10])
            fig.savefig(f'iter_{len(self.Xs)}.png', dpi=fig.dpi)
            # plt.show()









