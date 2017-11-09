from ddpg_tf import DDPG_tf
from ddpg_tflearn import DDPG_tflearn

class Brains():
    def __init__(
        self,
        sess,
        env_params,
        actor_lr=0.001,
        critic_lr=0.0001,
        tau=0.001,
        gamma=0.99
        ):
        self.brains = {
            'ddpg-tf': DDPG_tf(
                sess,
                env_params,
                actor_lr,
                critic_lr,
                tau,
                gamma
            ),

        }
    
    def get(self, brain):
        if brain in self.brains:
            return self.brains[brain]
        else:
            raise NotImplementedError('The brain "%s" was not found. ' 
                                        'Please try a different brain.' % brain)