import numpy as np
import random

class ExperienceReplay():
    """Stores past experience and replays random samples"""
    def __init__(self, memory_params):
        random.seed(1234)
        self.capacity = memory_params['capacity']
        self.minibatch_size = memory_params['minibatch_size']
        self.samples = []
        self.size = 0

    def add(self, experience):
        self.samples.append(experience)

        if self.size > self.capacity:
            self.samples.pop(0)
        else:
            self.size += 1
    
    def sample(self):
        n = min(self.minibatch_size, len(self.samples))
        batch = random.sample(self.samples, n)

        states = np.array([o[0] for o in batch])
        actions = np.array([o[1] for o in batch])
        rewards = np.array([o[2] for o in batch])
        states_ = np.array([o[3] for o in batch])
        stops = np.array([o[4] for o in batch])

        return states, actions, rewards, states_, stops, n
