import numpy as np
import time
import math
from scipy.spatial.distance import pdist, cdist, squareform
from render import *
import random

from gym import spaces
from physics import PhysicsEnvironment
from blueprints import Blueprints


class MicroRobot():
    def __init__(
            self,
            bounds=[0, 10, 0, 10],
            particle_size=0.1,
            agent_size=0.1,
            particle_mass=0.5,
            particle_density=7850,
            agent_mass=1,
            fluid_viscosity=1,
            fluid_density=1420,
            dt=0.05,
            preset='random',
            continuous=True,
            noise=False
        ):

        # Reward dictionary
        self.rewards = {
            'step': -1,
            'approach_target': 1,
            'touch_target': 100,
            'approach_goal': 90,
            'goal': 1000,
            'touch_obstacle': -10
        }

        # Load blueprints
        blueprints = Blueprints()
        self.bp_library = blueprints.library
        bp_key = random.choice(list(self.bp_library))
        self.blueprint = self.bp_library[bp_key]

        # randomly choose a template from the blueprint library
        self.particle_num = len(self.blueprint)

        # Setup physics environment
        self.phys_env = PhysicsEnvironment(
            self.particle_num+1,
            particle_mass,
            agent_mass,
            particle_size,
            agent_size,
            bounds,
            fluid_viscosity,
            fluid_density
        )

        self.episode = 0
        self.time_elapsed = 0
        self.add_noise = noise
        self.is_continuous = continuous
        self.dt = dt
        self.isrendered = False

        self.min_action = -5.
        self.max_action = 5.

        self.min_position_x = 0.
        self.min_position_y = 0.
        self.min_vel_x = -5.
        self.min_vel_y = -5.
        self.min_goal_pos_x = 0.
        self.min_goal_pos_y = 0.
        self.min_obstacle_x = 0.
        self.min_obstacle_y = 0.
    
        self.max_position_x = 10.
        self.max_position_y = 10.
        self.max_vel_x = -5.
        self.max_vel_y = -5.
        self.max_goal_pos_x = 10.
        self.max_goal_pos_y = 10.
        self.max_obstacle_x = 10.
        self.max_obstacle_y = 10.

        self.min_state = np.array([
            self.min_vel_x, # agent
            self.min_vel_y, # agent
            self.min_position_x, # target
            self.min_position_y, # target
            self.min_vel_x, # target
            self.min_vel_y, # target
            self.min_goal_pos_x,
            self.min_goal_pos_y,
            self.min_obstacle_x,
            self.min_obstacle_y
        ])

        self.max_state = np.array([
            self.max_vel_x,
            self.max_vel_y,
            self.max_position_x, # target
            self.max_position_y, # target
            self.max_vel_x, # target
            self.max_vel_y, # target
            self.max_goal_pos_x,
            self.max_goal_pos_y,
            self.max_obstacle_x,
            self.max_obstacle_y
        ])

        self.action_space = spaces.Box(self.min_action, self.max_action, (2,))
        self.observation_space = spaces.Box(self.min_state, self.max_state)

        self.preset = preset
        np.random.seed(1)

        # reset the environment
        self.reset()

        # initialize visualization
        self.init_visualization(agent_size, particle_size)

        ## Temporarily disabled magnetic field calculations ##
        # self.magfield = 0
        # self.Fx_buff = 0
        # self.Fy_buff = 0
        # self.Fx_out = 0
        # self.Fy_out = 0

        """
        states:             [target, part, part, part...]
        target_goal_list:   [(p, g), (p, g), ...]
        not_reached:        [1,2,3,4...] index of target_goal_list
        reached:            [1,2,3,4...] index of target_goal_list
        """

    def _init_classes(self, N, class_num):
        self.p_class = np.zeros(N+1).astype(int)
        indices = list(range(N+1))
        splits = np.array_split(indices, class_num)

        for i, split in enumerate(splits):
            self.p_class[split] = i

    def reset(self):
        self.time_elapsed = 0.
        self.total_reward = 0
        self.total_goals = 0
        self.steps = 0
        self.allowed_time = 30.
        self.done = False
        self.reached = []
        self.not_reached = []

        bp_key = random.choice(list(self.bp_library))
        self.blueprint = self.bp_library[bp_key]

        state, agent_dists = self.phys_env.reset()
        self.p_class = np.zeros(self.particle_num+1).astype(int)
        self.assign_targets(state)
        self.check_at_goal(state)
        self.pick_new_target()
        # self.pick_target(state)
        obs = self.get_observations(state)

        return obs

    def dac(self, arr):
        
        
        min_in_l = np.min(distance_arr[:(p_ix-1)])
        
    def assign_targets(self, state):
        particles = state[1:, :2]
        
        self.not_reached = list(range(len(particles)))
        self.reached = []
        self.target_goal_list = []
        distance_arr = cdist(particles, self.blueprint)

        while np.min(distance_arr) < np.inf:
            p_ix, g_ix = np.unravel_index(np.argmin(distance_arr), distance_arr.shape)
            self.target_goal_list.append((p_ix+1, g_ix))
            # set row to inf
            distance_arr[p_ix, :] = np.inf
            # remove col to inf
            distance_arr[:, g_ix] = np.inf
    
    def is_at_goal(self, target_id, goal_id, state):
        target = np.array([state[target_id, :2]])
        goal = np.array([self.blueprint[goal_id]])
        mostly_still = abs(np.sum(state[target_id,2:])) < 0.01
        close_to_goal = float(cdist(target, goal)) < 0.2
        return close_to_goal, mostly_still

    def check_at_goal(self, state):
        self.reached = []
        self.not_reached = []
        self.goals_reached = []
        for i, t_g in enumerate(self.target_goal_list):
            t, g = t_g
            target_at_goal, _ = self.is_at_goal(t, g, state)
            if target_at_goal:
                self.reached.append(i)
                self.goals_reached.append(g)
                self.p_class[t] = 1
            else:
                self.not_reached.append(i)
                self.p_class[t] = 0
        if len(self.not_reached) == 0:
            self.done = True

    def get_closest_obstacle(self, state):
        agent = state[0,:2]

        # add walls as obstacles
        # obstacles = np.array(
        #     [[agent[0],0.],
        #     [agent[0], 10.],
        #     [0.,agent[1]],
        #     [10.,agent[1]]]
        # )
        obstacles = np.array([[10,10]])

        # add targets at goals reached as obstacles
        for r in self.reached:
            p_id = self.target_goal_list[r][0]
            obstacles = np.vstack((obstacles, state[p_id,:2]))

        # get distances
        closest_obstacle = np.array([1.,1.])
        distances = cdist(np.array([agent]), obstacles)
        d_min = np.min(distances)
        closest_obstacle = obstacles[np.argmin(distances)]

        return closest_obstacle, d_min


    def pick_new_target(self):
        self.target_id, self.goal_id = self.target_goal_list[np.random.choice(self.not_reached)]

    def get_reward(self, state, agent_dists, wall_touch):
        agent = state[0,:2]
        target = state[self.target_id,:2]
        goal = self.blueprint[self.goal_id,:]
        d_to_target = round(agent_dists[self.target_id],2)
        d_to_goal = round(float(cdist(np.array([target]), np.array([goal]))),1)
        _, d_to_obstacle = self.get_closest_obstacle(state)

        if self.steps == 1:
            self.prev_d_to_goal = d_to_goal
            self.prev_d_to_target = d_to_target
            self.untouched = True

        # PENALTY FOR TAKING A STEP
        reward = self.rewards['step']

        # BEFORE TARGET TOUCH
        if d_to_target < self.prev_d_to_target and self.untouched:
            reward = self.rewards['approach_target']
            # print('approaching target')

        # TARGET TOUCH
        if d_to_target < 0.1 and self.untouched:
            if self.untouched:
                print('Touch!')
                reward = self.rewards['touch_target']
                self.untouched = False

        # GO TO GOAL
        if d_to_goal < self.prev_d_to_goal:
            # print('approaching goal')
            reward = self.rewards['approach_goal']

        # REACH GOAL
        # if close to goal and no longer moving
        at_goal, mostly_still = self.is_at_goal(self.target_id, self.goal_id, state)
        if at_goal and mostly_still:
            reward = self.rewards['goal']
            print('GOAL for %d!' % self.goal_id)
            self.info = 'success'
            self.check_at_goal(state)
            self.pick_new_target()
            self.allowed_time += 15.
            self.total_goals += 1
            self.untouched = True
        
        # self.moved_from_goal()

        if d_to_obstacle < 0.2:
            reward = self.rewards['touch_obstacle']

        self.prev_d_to_goal = d_to_goal
        self.prev_d_to_target = d_to_target

        return reward
    
    def step(self, action):
        self.info = ''
        action = action.flatten()
        dt = 0.05
        t = 0
        self.time_elapsed += dt
        self.steps += 1

        if self.is_continuous:
            Fx = action[0]
            Fy = action[1]
        else:
            actions = [-5., 5., -5., 5.]
            Fx = 0.
            Fy = 0.

            if action < 2:
                Fx = actions[action]
            else:
                Fy = actions[action]
            
        
        # run the physics environment in small intermediate steps
        # this is to resolve overlaps in fast-moving particles
        for _ in range(int(dt/self.phys_env.interval_dt)):
            state, done, agent_dists, wall_touch = self.phys_env.step(Fx, Fy)
            if done:
                break

        reward = self.get_reward(state, agent_dists, wall_touch)
        obs = self.get_observations(state)

        self.total_reward += reward

        if self.time_elapsed > self.allowed_time:
            self.done = True
        
        if self.render:
            self.update_visualization(state)
        
        if self.done:
            self.episode += 1
            self.update_visualization_stats(self.episode, self.total_reward, self.total_goals)
        # print(reward)
        return obs, reward, self.done, self.info

    def get_observations(self, state):
        self.check_at_goal(state)
        if self.done:
            s = self.min_state
            return None

        closest_obstacle, _ = self.get_closest_obstacle(state)
        agent_pos = state[0,:2]
        s = state[0,2:] # agent vel (shape=[2,])
        # s = np.append(s, state[0,2])
        s = np.append(s, state[self.target_id,:]) # target pos and vel (shape=[6,])
        s = np.append(s, self.blueprint[self.goal_id,:]) # append target goal pos (shape=[8,])
        s = np.append(s, closest_obstacle) # append closest obstacle pos (shape=[10,])
        # s = np.append(s, agent_pos)

        # make all positions relative to agent position, clamped between -1. and 1.

        # s[6] -= s[2]
        # s[7] -= s[3]
        
        # s[2] -= agent_pos[0]
        # s[3] -= agent_pos[1]
        # s[8] -= agent_pos[0]
        # s[9] -= agent_pos[1]

        s[6] = np.clip(s[6] - s[2], -2., 2.)
        s[7] = np.clip(s[7] - s[3], -2., 2.)
        
        s[2] = np.clip(s[2] - agent_pos[0], -2., 2.)
        s[3] = np.clip(s[3] - agent_pos[1], -2., 2.)
        # s[8] = np.clip(s[8] - agent_pos[0], -2., 2.)
        # s[9] = np.clip(s[9] - agent_pos[1], -2., 2.)
        s[8] = 1.
        s[9] = 1.
        if self.add_noise:
            s += np.random.normal(0.0, self.variance, (9))
        # print(s)
        # print("target: {:.2f},{:.2f}  goal: {:.2f},{:.2f}  obstacle: {:.2f},{:.2f}".format(s[2], s[3], s[6], s[7], s[8], s[9]))
        return s

    def init_visualization(self, agent_size, particle_size):
        self.vis_data = {
            'env': 'microbot-v1',
            'agent_size': agent_size,
            'particle_size': particle_size,
            'scores': dict(x=[0], y=[0]),
            'reward': 0,
            'reward_smoothed': 0,
            'episode': 0,
            'goal_count': 0,
            'goals_smoothed': 0,
            'particles': dict(x=[0], y=[0], p_class=[0]),
            'blueprint': dict(x=[0], y=[0]),
            'goal': dict(x=[0], y=[0]),
            'completed': dict(x=[0], y=[0]),
            'agents': dict(x=[0], y=[0]),
            'target': dict(x=[0], y=[0], p_class=[0])
        }


    def update_visualization(self, state):
        self.vis_data['particles'] = dict(x=state[1:,0], y=state[1:,1], p_class=self.p_class[1:])
        self.vis_data['agents'] = dict(x=[state[0,0]], y=[state[0,1]])
        self.vis_data['target'] = dict(x=[state[self.target_id,0]], y=[state[self.target_id,1]], 
                                       p_class=[self.p_class[self.target_id]])
        self.vis_data['blueprint'] = dict(x=self.blueprint[:,0], y=self.blueprint[:,1])
        self.vis_data['goal'] = dict(x=[self.blueprint[self.goal_id,0]], y=[self.blueprint[self.goal_id,1]])
        self.vis_data['completed'] = dict(x=[self.blueprint[self.goals_reached,0]], 
                                            y=[self.blueprint[self.goals_reached,1]])
    
    def update_visualization_stats(self, episode, reward, goals):
        reward_smoothed = 0.01*reward + 0.99*self.vis_data['reward_smoothed']
        goals_smoothed = 0.01*goals + 0.99*self.vis_data['goals_smoothed']
        self.vis_data['reward'] = reward
        self.vis_data['reward_smoothed'] = reward_smoothed
        self.vis_data['episode'] = episode
        self.vis_data['goal_count'] = goals
        self.vis_data['goals_smoothed'] = goals_smoothed
        
    def toggle_speed(self):
        if self.isrendered:
            self.isrendered = False
        else:
            self.isrendered = True

    def render(self):
        self.renderer = Render(self.vis_data)
