import numpy as np
import random

class Agent():
    """
    Agent which can act, observe and replay past experiences.
    Requires a memory for experience storage and replay and
    a brain for generating actions given a state
    """
    def __init__(self, memory, brain, env_params, exploration_params):
        self.memory = memory
        self.brain = brain

        self.env_params = env_params
        self.exp_params = exploration_params

        self.epsilon = self.exp_params['exploration_init']
        self.epsilon_final = self.exp_params['exploration_final']
        self.epsilon_decay = self.exp_params['exploration_rate']
        self.noise = OrnsteinUhlenbeckActionNoise(env_params)

    def act(self, state):
        """
        Generate an action given a state.

        Two types of exploration are handled. In the 'random'
        case, actions are generated randomly with probability
        epsilon. In the 'noise' case, noise is added to the
        predicted action.

        Arguments:
            state: state vector of shape=[1, state_dim]
        
        Returns:
            action: action predicted by brain
        """
        a = self.brain.predict(state)

        if self.exp_params['exploration_type'] == 'random' and random.random() < self.epsilon:
            a = self.explore()

        elif self.exp_params['exploration_type'] == 'noise':
            if self.env_params['env_type'] == 'discrete':
                raise NotImplementedError('You may not use noisy exploration in conjunction '
                                        'with a discrete action space. Please choose a different '
                                        'exploration-type')
            a += self.noise()

        return a

    def explore(self):
        if self.env_params['env_type'] == 'discrete':
            a = np.random.randint(0, self.env_params['action_dim'])
        else:
            a = np.random.uniform(self.env_params['action_bounds'][0], 
                                self.env_params['action_bounds'][1],
                                (self.env_params['action_dim'],))
        
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay
        return a

    def a3c_train(self, experience):
        """
        for use in A3C
        """
        self.observe(experience)

        if self.memory.size >= N_STEP_RETURN:
            s, a, r, s_, done, batch_size = self.memory.sample(1)


    def observe(self, experience):
        """
        Store an experience for future replay.

        Arguments:
            experience: (s, a, r, s_, stop)
        """
        self.memory.add(experience)

    def update(self):
        self.brain.update()
    
    def replay(self):
        """
        Retrieves a batch of randomly choseen past experiences and
        replays them to update target networks.

        Returns:
            max_q: maximum predicted q value for the batch
        """
        states, actions, rewards, states_, stops, batch_size = self.memory.sample()
        # if batch_size == 64:
        max_q = self.brain.update_targets(states, actions, rewards, states_, stops, batch_size)
        
        return max_q

class OrnsteinUhlenbeckActionNoise:
    """
    Noise generation for exploration purposes.
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Credit: Patrick Emami
    """
    def __init__(self, env_params, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.mu=np.zeros(env_params['action_dim'])
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
