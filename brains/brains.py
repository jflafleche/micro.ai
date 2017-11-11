class Brains():
    def __init__(
        self,
        sess,
        env_params,
        brain_params
        ):
        self.sess = sess
        self.env_params = env_params
        self.brain_params = brain_params
        self.brains = [
            'ddpg-tf', 
            'duelling'
            ]
    
    def get(self, brain):
        if brain in self.brains:
            if brain == 'ddpg-tf':
                from ddpg_tf import DDPG_tf
                return DDPG_tf(self.sess, self.env_params, self.brain_params)

            elif brain == 'duelling':
                from duelling import DuellingDQN
                return DuellingDQN(self.sess, self.env_params, self.brain_params)
        else:
            raise NotImplementedError('The brain "%s" was not found. ' 
                                        'Please try a different brain.' % brain)

    def import_from(self, module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)