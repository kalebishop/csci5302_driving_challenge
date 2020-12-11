import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        # action took from parent to get to current state
        self.speed = 0
        self.steering = 0


class Obstacle:
    def __init__(self, position, x_length, y_length):
        self.position = position
        self.x_length = x_length
        self.y_length = y_length

    def get_boundary_positions(self):
        """
        Returns the positions of the four corners of the obstacle
        """
        offset_x = self.x_length/2.0
        x, y = self.position
        return [[x-offset_x, y-self.y_length*1/5.0],
            [x-offset_x, y+self.y_length*4/5.0],
            [x+offset_x, y+self.y_length*4/5.0],
            [x+offset_x, y-self.y_length*1/5.0]]


class RRT:
    def __init__(self):
        self.start_state = [87, 159, 0]
        self.goal_state = [80, 143, 0]
        self.x_length = 1.8
        self.y_length = 4.5
        self.obstacles = [
            Obstacle([80, 147], 2, 7),
            Obstacle([80, 134], 2, 5)
        ]
        # road width = 17
        self.state_bounds = [[77, 88], [127, 170]]
        self.sample_space = [[78, 85], [130, 150]]
        # self.state_bounds = x_bounds, y_bounds
        self.thetas = np.arange(0, 360, 10) * np.pi / 180.
        self.axle_length = 2.875
        # simulation time step in seconds
        self.time_step = 0.01
        self.goal_bias = 0.2

    def wrap_angle(self, angle):
        """
        Wrap angle to [-pi, pi]
        :param angle: angle in radians
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_boundary_positions(self, state):
        """
        Returns the positions of the boundaries of the car
        """
        half_x = self.x_length/2.0
        offset_y_neg = self.y_length*4/5.0
        offset_y_pos = self.y_length*1/5.0
        x, y, theta = state
        boundary = np.array([[x-half_x, y-offset_y_neg], [x-half_x, y], [x-half_x, y+offset_y_pos],
                        [x, y+offset_y_pos], [x+half_x, y+offset_y_pos], [x+half_x, y],
                        [x+half_x, y-offset_y_neg], [x, y-offset_y_neg]])
        # theta is positive clockwise, so negate theta in rotation matrix
        rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)]])
        boundary = np.matmul(rotation_matrix, boundary.T).T
        return boundary

    def sample_random_point(self):
        sampled_pos = [np.random.uniform(dim[0], dim[1]) for dim in self.sample_space]
        sampled_theta = [np.random.choice(self.thetas)]
        return np.array(sampled_pos + sampled_theta)

    def get_distance(self, point1, point2):
        pos_dist = np.linalg.norm(np.array(point1[:2])-np.array(point2[:2]))
        theta_dist = np.arccos(np.cos(point1[2]-point2[2]))
        return pos_dist * 2.0 + theta_dist

    def get_nearest_node(self, node_list, sampled_point):
        closest_node = None
        closest_dist = np.inf
        for node in node_list:
            dist = self.get_distance(node.point, sampled_point)
            if dist < closest_dist:
                closest_dist = dist
                closest_node = node
        return closest_node, closest_dist

    def state_is_valid(self, state):
        # TODO: need to check more than the four corners
        boundary = self.get_boundary_positions(state)
        for dim in range(len(self.state_bounds)):
            if state[dim] <= self.state_bounds[dim][0]: return False
            if state[dim] >= self.state_bounds[dim][1]: return False

        for obs in self.obstacles:
            x_min = obs.position[0] - obs.x_length*1/5.0
            x_max = obs.position[0] + obs.x_length*4/5.0
            y_min = obs.position[1] - obs.y_length*1/5.0
            y_max = obs.position[1] + obs.y_length*4/5.0
            if x_min <= state[0] <= x_max and y_min <= state[1] <= y_max:
                return False
            for b in boundary:
                if x_min <= b[0] <= x_max and y_min <= b[1] <= y_max:
                    return False
        return True

    def collision_free(self, path):
        if path is None:
            return False
        valid_path = True
        for state in path:
            if not self.state_is_valid(state):
                valid_path = False
                break
        return valid_path

    def goal_reached(self, state):
        # within one meter in x direction
        if abs(state[0] - self.goal_state[0]) > 1:
            return False
        # within one meter in y direction
        if state[1] < self.goal_state[1] - 1 or state[1] > self.goal_state[1] + 1:
            return False
        # with in 10 degrees of 0 or 180
        bound = 10*np.pi/180.
        if -bound < state[2] < bound or 170*np.pi/180. < state[2] < 190*np.pi/180.:
            return True
        return False

    def steer(self, current_node, target_speed, steering_angle):
        state = current_node.point
        path = [state]

        speed = target_speed * (1000 / 3600.) * self.time_step
        # perform action for some steps
        for i in range(100):
            x, y, theta = state

            yp = speed * np.cos(theta)
            xp = speed * np.sin(theta)
            # self.axle length is distance btw axles
            thetap = (speed / self.axle_length) * np.tan(steering_angle)

            next_state = [x+xp, y+yp, self.wrap_angle(theta+thetap)]
            path.append(next_state)
            current_speed = speed
            state = next_state
        return path

    def generate_graph(self, max_nodes=300):
        node_list = [Node(self.start_state)]
        goal_found = False

        while len(node_list) < max_nodes and not goal_found:
            if np.random.uniform() < self.goal_bias:
                sampled_point = self.goal_state
            else:
                sampled_point = self.sample_random_point()
            neighbor, dist = self.get_nearest_node(node_list, sampled_point)
            # sample different actions (target_speed and steering_angle)
            best_action = None
            closest_distance = np.inf
            for speed in np.linspace(-10, 10, 10):
                for steering_angle in np.linspace(-0.7, 0.7, 15):
                    path = self.steer(neighbor, speed, steering_angle)

                    if not self.collision_free(path):
                        continue

                    dist = self.get_distance(sampled_point, path[-1])

                    if dist < closest_distance:
                        closest_distance = dist
                        best_action = (speed, steering_angle, path, dist)

            if best_action is not None:
                speed, steering_angle, path, dist = best_action
                sampled_node = Node(path[-1], neighbor)
                sampled_node.speed = speed
                sampled_node.steering = steering_angle
                sampled_node.path_from_parent = path
                node_list.append(sampled_node)
                goal_found = self.goal_reached(path[-1])
        print("Goal found: " + str(goal_found))
        return node_list

    def get_actions(self, node_list):
        node, dist = self.get_nearest_node(node_list, self.goal_state)
        actions = []

        while node.parent is not None:
            actions.append((node.speed, node.steering))
            node = node.parent
        return actions[::-1]

    def visualize_2D_graph(self, node_list, goal_node=None):
        fig = plt.figure()
        plt.xlim(self.state_bounds[0][0], self.state_bounds[0][1])
        plt.ylim(self.state_bounds[1][0], self.state_bounds[1][1])

        for obs in self.obstacles:
            boundary = obs.get_boundary_positions()
            obs_x, obs_y = zip(*boundary)
            obs_x, obs_y = list(obs_x), list(obs_y)
            obs_x.append(obs_x[0])
            obs_y.append(obs_y[0])
            plt.plot(obs_x, obs_y)

        for node in node_list:
            if node.parent is not None:
                node_path = np.array(node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '-b')
            plt.plot(node.point[0], node.point[1], 'ro')

        if goal_node is not None:
            curr_node = goal_node
            while curr_node is not None:
                if curr_node.parent is not None:
                    node_path = np.array(curr_node.path_from_parent)
                    plt.plot(node_path[:,0], node_path[:,1], '--y')
                    curr_node = curr_node.parent
                else:
                    break

        plt.plot(node_list[0].point[0], node_list[0].point[1], 'ko')
        plt.plot(self.goal_state[0], self.goal_state[1], 'gx')
        fig.savefig("rrt2D.png")
        plt.show()

    def visualize_3D_graph(self, node_list, goal_node=None):
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set(xlabel='x',
               ylabel='y',
               zlabel='theta',
               aspect='auto',
               xlim3d=(self.state_bounds[0][0], self.state_bounds[0][1]),
               ylim3d=(self.state_bounds[1][0], self.state_bounds[1][1]),
               zlim3d=(-np.pi, np.pi))

        # plot nodes
        node_positions = np.array(list(map(lambda x: x.point, node_list)))
        x, y, theta = zip(*node_positions)
        ax.scatter3D(x, y, theta, 'ro')

        # plot edges
        for node in node_list:
            if node.parent is not None:
                node_path = np.array(node.path_from_parent)
                ax.plot3D(node_path[:,0], node_path[:,1], node_path[:,2], '-b')

        if goal_node is not None:
            curr_node = goal_node
            while curr_node is not None:
                if curr_node.parent is not None:
                    node_path = np.array(curr_node.path_from_parent)
                    ax.plot3D(node_path[:,0], node_path[:,1], node_path[:,2], '--y')
                    curr_node = curr_node.parent
                else:
                    break

        # plot obstacles
        for theta in range(-180, 180, 10):
            for obs in self.obstacles:
                boundary = obs.get_boundary_positions()
                obs_x, obs_y = zip(*boundary)
                thetas = [theta*np.pi/180.] * len(boundary)
                verts = [list(zip(obs_x, obs_y, thetas))]
                poly = Poly3DCollection(verts)
                poly.set_alpha(0.15)
                poly.set_facecolor('w')
                poly.set_edgecolor('r')
                ax.add_collection3d(poly)

        ax.plot3D([self.start_state[0]], [self.start_state[1]], [0], 'ko')
        ax.plot3D([self.goal_state[0]], [self.goal_state[1]], [0], 'gx')
        fig.savefig("rrt3D.png")
        plt.show()


def main():
    rrt = RRT()
    node_list = rrt.generate_graph()
    actions = rrt.get_actions(node_list)
    print(actions)
    best_node, dist = rrt.get_nearest_node(node_list, rrt.goal_state)
    print("Nearest node: " + str(best_node.point))
    rrt.visualize_2D_graph(node_list, best_node)
    rrt.visualize_3D_graph(node_list, best_node)


if __name__ == "__main__":
    main()
