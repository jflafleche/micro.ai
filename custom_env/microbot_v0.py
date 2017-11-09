import numpy as np
import time
import math
from scipy.spatial.distance import pdist, cdist, squareform
from render import *
import random

from gym import spaces
from physics import PhysicsEnvironment

np.random.seed(1234)

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
            continuous=True,
            noise=False
        ):
        # number of particles, scenario dependant
        self.particle_num = 10

        self.rewards = {
            'step': -1,
            'approach_target': 1,
            'touch_target': 10,
            'approach_goal': 90,
            'goal': 100,
            'wall_touch': -1
        }
        
        # setup physics environment
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
        self.min_class = 0
        self.max_position_x = 0.
        self.max_position_y = 0.
        self.max_vel_x = -5.
        self.max_vel_y = -5.
        self.max_class = int(self.particle_num // 2)

        self.min_state = np.array([
            self.min_position_x,
            self.min_position_y,
            self.min_vel_x,
            self.min_vel_y,
            self.min_position_x, # target
            self.min_position_y, # target
            self.min_vel_x, # target
            self.min_vel_y, # target
            self.min_class # target class
        ])

        self.max_state = np.array([
            self.max_position_x,
            self.max_position_y,
            self.max_vel_x,
            self.max_vel_y,
            self.max_position_x, # target
            self.max_position_y, # target
            self.max_vel_x, # target
            self.max_vel_y, # target
            self.max_class # target class
        ])

        self.action_space = spaces.Box(self.min_action, self.max_action, (2,))
        self.observation_space = spaces.Box(self.min_state, self.max_state)
        
        self.reset()
        self.init_visualization(agent_size, particle_size)
        
        # self.magfield = 0
        # self.Fx_buff = 0
        # self.Fy_buff = 0
        # self.Fx_out = 0
        # self.Fy_out = 0
        
        # self.variance = agent_size
        # self.time_list = []

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
        self.done = False
        self.allowed_time = 30.

        state, agent_dists = self.phys_env.reset()

        # setup classes
        self._init_classes(self.particle_num, 2)

        self.Q = list(range(1, self.particle_num+1))
        self.pick_target(state, agent_dists)

        obs = self.get_observations(state)

        return obs
    
    def pick_target(self, state, dists):
        if len(self.Q) > 0:
            target_dists = dists[self.Q]
            self.target_id = self.Q[np.argmin(target_dists)]
            self.target_class = self.p_class[self.target_id]

            # update queue
            self.Q.pop(self.Q.index(self.target_id))
        
        else:
            self.done = True
    
    def get_reward(self, state, agent_dists, wall_touch):
        reward = self.rewards['step'] # penalty for taking a step
        d_to_target = round(agent_dists[self.target_id],2)

        if self.steps == 1:
            self.prev_d = d_to_target
            self.prev_d_to_goal = 0.
        
        if self.target_class == 1:
            goal = 9.0
            goal_dir = 1
        else:
            goal = 1.0
            goal_dir = -1 # neg sign to indicate goal pos is to the left of 1.0

        target_pos_x = round(state[self.target_id, 0], 1)
        d_to_goal = goal_dir*(target_pos_x - goal)
        goal = d_to_goal > 0

        if self.steps == 1:
            prev_d_to_goal = d_to_goal
            self.prev_d = d_to_target
            self.untouched = True

        # if agent has not yet touched the target
        if self.untouched:
            # apprach reward
            if d_to_target < self.prev_d:
                reward = self.rewards['approach_target']
                
            # if cdist(agent, target) < 0.21 and self.untouched:
            if d_to_target < 0.1:
                print('TOUCH!')
                reward = self.rewards['touch_target']
                self.untouched = False

        if abs(d_to_goal) < self.prev_d_to_goal and not self.untouched:
            # print('getting closer to goal')
            reward = self.rewards['approach_goal']
        
        if wall_touch:
            reward = self.rewards['wall_touch']

        if goal:
            print('GOAL for %d!' % self.target_id)
            self.info = 'success'
            self.allowed_time += 30.
            reward = self.rewards['goal']
            self.total_goals += 1
            self.untouched = True
            self.pick_target(state, agent_dists)

        self.prev_d_to_goal = abs(d_to_goal)
        self.prev_d = d_to_target

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

        if self.time_elapsed > self.allowed_time:
            self.done = True
        
        if self.render:
            self.update_visualization(state)
        
        if self.done:
            self.episode += 1
            self.update_visualization_stats(self.episode, self.total_reward, self.total_goals)

        return obs, reward, self.done, self.info

    def get_observations(self, state):
        s = state[[0,self.target_id],:]
        s = np.append(s,self.target_class)
        s[4] -= s[0]
        s[5] -= s[1]

        # if self.add_noise:
        #     s += np.random.normal(0.0, self.variance, (9))

        return s

    def init_visualization(self, agent_size, particle_size):
        self.episode = 0

        self.vis_data = {
            'env': 'microbot-v0',
            'agent_size': agent_size,
            'particle_size': particle_size,
            'scores': dict(x=[0], y=[0]),
            'reward': 0,
            'reward_smoothed': 0,
            'episode': 0,
            'goals': 0,
            'goals_smoothed': 0,
            'trackdots': dict(x=[0], y=[0]),
            'particles': dict(x=[0], y=[0], p_class=[0]),
            'render_blueprint_dict': dict(x=[0], y=[0]),
            'render_goal_dict': dict(x=[0], y=[0]),
            'render_completed_dict': dict(x=[0], y=[0]),
            'agents': dict(x=[0], y=[0]),
            'target': dict(x=[0], y=[0], p_class=[0])
        }


    def update_visualization(self, state):
        self.vis_data['particles'] = dict(x=state[1:,0], y=state[1:,1], p_class=self.p_class[1:])
        self.vis_data['agents'] = dict(x=[state[0,0]], y=[state[0,1]])
        self.vis_data['target'] = dict(x=[state[self.target_id,0]], y=[state[self.target_id,1]], 
                                        p_class=[self.p_class[self.target_id]])
    
    def update_visualization_stats(self, episode, reward, goals):
        reward_smoothed = 0.01*reward + 0.99*self.vis_data['reward_smoothed']
        goals_smoothed = 0.01*goals + 0.99*self.vis_data['goals_smoothed']
        self.vis_data['reward'] = reward
        self.vis_data['reward_smoothed'] = reward_smoothed
        self.vis_data['episode'] = episode
        self.vis_data['goal_count'] = goals
        self.vis_data['goals_smoothed'] = goals_smoothed
        
        # if self.isrendered:
        #     time.sleep(self.iterdt)
        # print(self.Fx)
        
    def toggle_speed(self):
        if self.isrendered:
            self.isrendered = False
        else:
            self.isrendered = True

    def render(self):
        self.renderer = Render(self.vis_data)

    def clamp(self, val, upbound, lobound):
        if val > upbound:
            return upbound
        elif val < lobound:
            return lobound
        else:
            return val
