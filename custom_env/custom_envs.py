import microbot_v0
import microbot_v1

class CustomEnvs():
    def __init__(self):
        self.envs = {
            'microbot-v0': microbot_v0.MicroRobot(),
            'microbot-v1': microbot_v1.MicroRobot(),
        }