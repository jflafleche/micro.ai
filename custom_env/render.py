# myapp.py
# from microbot import MicroRobot

import time
import math
from threading import Thread

from bokeh.client import push_session
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Button, LabelSet
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure, curdoc
from magfield import MagField

from functools import partial
from tornado import gen


class Render():
    def __init__(self, vis_data):
        self.episode = 0
        self.magfield = MagField()
        self.doc = curdoc()
        self.vis_data = vis_data
        self.current_ep = 0
        
        # open a session to keep local doc in sync with server
        self.session = push_session(self.doc)

        # p_colors = []
        # for c in vis_data['classes']:
        #     col = '#00BCD4' if c == 0 else '#E91E63'
        #     p_colors.append(col)

        self.p_colors = ['#00BCD4', '#E91E63']

        # setup sources
        particles_dict = self.vis_data['particles']
        particles_dict['colors'] = [self.p_colors[c] for c in particles_dict['p_class']]
        target_dict = self.vis_data['target']
        target_dict['colors'] = [self.p_colors[c] for c in target_dict['p_class']]
        self.particle_source = ColumnDataSource(data=particles_dict)
        self.agent_source = ColumnDataSource(data=vis_data['agents'])
        self.reward_source = ColumnDataSource({'episodes': [], 'rewards': [], 'rewards_smoothed': []})
        self.goal_count_source = ColumnDataSource({'episodes': [], 'goal_count': [], 'goals_smoothed': []})
        self.target_source = ColumnDataSource(data=target_dict)
        
        if self.vis_data['env'] == 'microbot-v1':
            self.blueprints_source = ColumnDataSource(data=vis_data['blueprint'])
            self.goal_source = ColumnDataSource(data=vis_data['goal'])
            self.completed_source = ColumnDataSource(data=vis_data['completed'])

        # print(self.blueprints_source.data)
        # print(self.goal_source.data)
        # print(self.completed_source.data)

        # setup figures
        p = figure(plot_width=600, plot_height=600, x_range=(0,10), y_range=(0,10), title='Deep-Q, Micro Robot')
        rewardp = figure(plot_width=600, plot_height=300, x_range=(0,10000), y_range=(0,200), title='Rewards')
        goalp = figure(plot_width=600, plot_height=300, x_range=(0,10000), y_range=(0,10), title='Goals Reached')
        p2 = figure()
        p2.xaxis.visible = False
        p2.xgrid.visible = False
        p2.yaxis.visible = False
        p2.ygrid.visible = False
        p3 = figure(plot_width=600, plot_height=100, x_range=(0,120), y_range=(0,100), tools=[])
        p3.xaxis.visible = False
        p3.xgrid.visible = False
        p3.yaxis.visible = False
        p3.ygrid.visible = False

        # Draw
        if self.vis_data['env'] == 'microbot-v0':
            p.patch(x=[0, 0, 1, 1], y=[0,10,10, 0], color='#00BCD4', alpha=0.1)
            p.patch(x=[9, 9, 10, 10], y=[0,10,10, 0], color='#E91E63', alpha=0.1)
            # self.p.circle('x', 'y', color='color', source=self.trackdots_source, radius=0.05)
        elif self.vis_data['env'] == 'microbot-v1':
            p.circle('x', 'y', source=self.blueprints_source, 
                     radius=vis_data['particle_size']*2, color='#00BCD4', alpha=0.2)
            p.circle('x', 'y', source=self.goal_source, 
                     radius=self.vis_data['particle_size']*2, color='#00BCD4', line_width=2, fill_alpha=0.2)
            p.circle('x', 'y', source=self.completed_source, 
                     radius=self.vis_data['particle_size']*2, color='#E91E63', line_width=2, fill_alpha=1.0)

        p.circle('x', 'y', color='colors', source=self.particle_source, radius=vis_data['particle_size'])
        p.circle('x', 'y', color='colors', source=self.target_source, radius=vis_data['particle_size']+0.1, 
                 line_width=2, fill_alpha=0.0)
        p.circle('x', 'y', source=self.agent_source, radius=vis_data['agent_size'], color='#212121')

        ## stats
        rewardp.line(source=self.reward_source, x='episodes', y='rewards', color='#b8f7ff')
        rewardp.line(source=self.reward_source, x='episodes', y='rewards_smoothed', color='#E91E63')
        goalp.line(source=self.goal_count_source, x='episodes', y='goal_count', color='#b8f7ff')
        goalp.line(source=self.goal_count_source, x='episodes', y='goals_smoothed', color='#E91E63')
        
        grid = gridplot([[p, p2], [p3], [rewardp, goalp]])
        self.doc.add_root(grid)
        self.doc.add_periodic_callback(self.update,50)
        self.session.show()

        thread = Thread(target=self.session.loop_until_closed)
        thread.daemon=True
        thread.start()

    def update(self):
        particles_dict = self.vis_data['particles']
        particles_dict['colors'] = [self.p_colors[c] for c in particles_dict['p_class']]
        target_dict = self.vis_data['target']
        target_dict['colors'] = [self.p_colors[c] for c in target_dict['p_class']]
        self.target_source.data = target_dict
        self.particle_source.data = particles_dict
        self.agent_source.data = self.vis_data['agents']
        # if self.vis_data['env'] == 'microbot-v0':
        #     self.trackdots_source.data = self.vis_data.trackdots
        if self.vis_data['env'] == 'microbot-v1':
            self.blueprints_source.data = self.vis_data['blueprint']
            self.goal_source.data = self.vis_data['goal']
            self.completed_source.data = self.vis_data['completed']
        
        # only append to graphs if it's a new episode
        if self.current_ep != self.vis_data['episode']:
            self.current_ep = self.vis_data['episode']
            self.reward_source.stream({
                'episodes': [self.vis_data['episode']], 
                'rewards': [self.vis_data['reward']],
                'rewards_smoothed': [self.vis_data['reward_smoothed']]
                }, 10000)
            self.goal_count_source.stream({
                'episodes': [self.vis_data['episode']], 
                'goal_count': [self.vis_data['goal_count']],
                'goals_smoothed': [self.vis_data['goals_smoothed']]
                }, 10000)