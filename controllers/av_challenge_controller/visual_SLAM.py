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
        self.Sigma = Sigma if Sigma is not None else np.full((2, 2), float('inf'))


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
        self.new_mu_sigma = 1  # TODO adjust
        self.new_landmark_weight = 0.9
        self.Xs = []
        self.Ys = []
        self.particles = [Particle()]
        self.resample_particles()

    def motion_model(self, x, u):
        """
        Given previous state x, and action u, return motion model
        denoted as g in code, and G is it's jacobian (first derivative with respect to state)
        :param x: state (x, y, theta)
        :param u: action, expects it in the for (u_speed, u_change_in_steering_angle)
        :return: g(x, u) and G
        """
        # Below is adapted from slide deck 33, slides 16 and 20
        x, y, theta = x
        us, ua = u  # speed and angle change
        xp = x + us * (-sin(theta) + sin(theta + ua))
        yp = y + us * (cos(theta) + -cos(theta + ua))
        thetap = theta + ua

        # G = np.eye(3)
        # G[0][2] = us * (-cos(theta) + cos(theta + ua))
        # G[0][1] = us * (-sin(theta) + sin(theta + ua))

        return np.array([xp, yp, thetap]) #, G

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

    def get_landmarks(self, p, zt):
        """
        Associate landmarks visible in observation zt with landmarks of particle k.
        If landmark in zt has not been seen before, return None for that one.
        :param p: Particle
        :param zt: observations
        :return: list of associated landmarks
        """
        #TODO define better threshold
        # threshold = 10

        # might be useful later if we need to do data association using distances
        # v_to_p = {}
        # p_to_v = {}
        #
        # js = []
        # distances = np.zeros((len(zt), len(p.landmarks)))
        # # visual landmarks
        # for i, v_landmark in enumerate(zt):
        #     v_landmark = np.array(v_landmark[:2]) + p.mu[:2]
        #     # If there's no landmarks which it could map to, it's a new landmark
        #     if len(p.landmarks) - len(v_to_p) <= 0:
        #         js.append((v_landmark, None))
        #         continue
        #
        #
        #     # predicted landmarks
        #     for k, p_landmark in enumerate(p.landmarks):
        #         distances[i, k] = np.linalg.norm(v_landmark - p_landmark.mu)
        #
        #     # for visual landmark i, find closest particle landmark k
        #     closest = np.argmin(distances[i])
        #
        #     # if distance to nearest is greater than threshold, it's a new landmark
        #     if distances[i, closest] > threshold:
        #         js.append((v_landmark, None))
        #         continue
        #
        #     # Another landmark has already paired with the closest landmark
        #     if closest in p_to_v:
        #         # Deal with this later if it becomes an issue
        #         print("Ugh got to deal with this")
        #     else:
        #         js.append((v_landmark, p.landmarks[closest]))
        #         v_to_p[i] = closest
        #         p_to_v[closest] = i
        #         continue

        js = []
        for v_landmark in zt:
            id_, pos = v_landmark
            pos = np.array(pos[:2]) + p.mu[:2]
            for p_landmark in p.landmarks:
                if id_ == p_landmark.id:
                    js.append((pos, p_landmark))
                    break
            else:
                js.append((pos, id_))

        return js

    def EKF_Update(self, j, l):
        """
        Apply EKF update to landmark l of particle p using visual landmark j
        """
        # landmark = j.landmarks[k]
        # z_t_pred, H = self.observation_model(j.mu, xt)
        H = np.ones((2,2))
        self.Q = np.ones((2,2))

        Q = np.matmul(np.matmul(H, l.Sigma), H.transpose()) + self.Q
        # print(Q)
        K = np.matmul(np.matmul(l.Sigma, H), np.linalg.pinv(Q))  # Kalman gain # TODO using pseudo inverse of Q because otherwise getting singular matrices -- not sure what that's about
        l.mu = l.mu + np.matmul(K, (j - l.mu))
        l.Sigma = np.matmul((np.identity(2) - np.matmul(K, H)), l.Sigma) # TODO @Kaleb was there a reason you used 4 originally?

        weight = np.linalg.norm(Q) ** (-0.5) * np.exp(-0.5 * np.matmul(np.matmul((j - l.mu).T, np.linalg.pinv(Q)), (j - l.mu)))

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
            old_p = np.random.choice(new_particles)
            new_mu = old_p.mu + np.random.normal(self.new_mu_sigma, size=3)
            new_p = Particle(new_mu)
            for lm in old_p.landmarks:  # TODO can probably be more efficient
                mu = lm.mu + np.random.normal(self.new_mu_sigma, size=2)
                Sigma = lm.Sigma + np.random.normal(self.new_mu_sigma, size=(2, 2))
                new_p.add_landmark(lm.id, mu, Sigma)

            new_particles.append(new_p)

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
        # TODO Currently cheating
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
                    new_landmark_mu = p.mu[:2] + j[0] # replace with p.mu[:2] + j[0] when we fix visual cheat
                    new_landmark_Sigma = np.full((2,2), 5) # TODO redefine 5 (which is just arbitrary covariance matrix value for new landmarks)
                    p.add_landmark(j[1], new_landmark_mu, new_landmark_Sigma)
                    w_k_j = self.new_landmark_weight
                else:
                    w_k_j = self.EKF_Update(j[0], j[1])
                w_k.append(w_k_j)

            if len(w_k) > 0:
                w_k = np.power(np.prod(w_k), 1. / len(w_k)) # take n_th root of the product of the probabilities
                p.update_weight(w_k)

        self.resample_particles()

        if len(self.Xs) % 500 == 0:
            print('yipee')
            fig = plt.figure()
            plt.plot(self.Xs, self.Ys)
            fig.savefig(f'iter_{len(self.Xs)}.png', dpi=fig.dpi)
            # plt.show()









