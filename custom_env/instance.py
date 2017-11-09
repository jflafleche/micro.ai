"""
Instance class for implementation of 
A3C. Based on the excellent blog post
https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
by Jaromír Janisch.
"""

import time

THREAD_DELAY = 1

class Instance():
    def __init__(self, env, agent):
        self.stop_signal = False
        self.env = env
        self.agent = agent

    def run(self):
        while not self.stop_signal:
            self.run_episode()
    
    def run_episode(self):
        s = self.env.reset()
        while True:
            time.sleep(THREAD_DELAY) # yield

            a = self.agent.act()

            s_, r, done, info = self.env.step(a)

            self.agent.train(s, a, r, s_, done)

            s = s_.copy()

            if done or self.stop_signal:
                break