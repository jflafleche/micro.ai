"""
This code implements a DDPG network.
Based on the paper by Lillicrap et al.
    https://arxiv.org/abs/1509.02971

Code is a modified version of the
implementation by Patrick Emami:
    http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: Jean-Francois Lafleche, 2017
"""

import numpy as np
import random
import tensorflow as tf

np.random.seed(1234)
random.seed(1234)

class DDPG_tf():
    """ DDPG implementation using TensorFlow """
    def __init__(
        self, 
        sess, 
        env_params,
        brain_params
    ):
        self.state_dim = env_params['state_dim']
        self.action_dim = env_params['action_dim']
        self.actor_lr = brain_params['lr_1']
        self.critic_lr = brain_params['lr_2']
        self.discount_factor = brain_params['gamma']
        self.tau = brain_params['tau'] # soft target update param

        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, env_params['action_bounds'],
                                  self.actor_lr, self.tau)

        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim,
                                    self.critic_lr, self.tau)

    def predict(self, state):
        """ 
        Predict an action given a state.

        Arguments:
            state: state vector of shape=[1, state_dim]

        Returns:
            action: action vector
        """
        return self.actor.predict(np.reshape(state, (1, self.actor.state_dim)))

    def update(self):
        """
        Update target networks
        """
        self.actor.update_target_network()
        self.critic.update_target_network()
    
    def update_targets(self, states, actions, rewards, states_, stops, batch_size):
        """
        Update target networks given a minibatch.

        Arguments:
            states: states array of shape=[minibatch, state_dim]
            actions: action array of shape=[minibatch, action_dim]
            rewards: reward array of shape=[minibatch, 1]
            states_: states_ (states prime) array of shape=[minibatch, state_dim]
            stops: vector of bools identifying states where a stop signal was received
            batch_size: integer identifying length of minibatch

        Returns:
            max_q: maximum predicted q value for the batch
        """
        # Calculate targets
        target_pi = self.actor.predict_target(states_)
        target_q = self.critic.predict_target(states_, target_pi)
        
        stops = stops[:, np.newaxis]
        rewards = rewards[:, np.newaxis]
        
        y = rewards + self.discount_factor*target_q*(-stops)
        predicted_q_value, _ = self.critic.train(states, actions, y)
        max_q = np.amax(predicted_q_value)

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(states)
        grads = self.critic.action_gradients(states, a_outs)
        self.actor.train(states, grads[0])

        self.actor.update_target_network()
        self.critic.update_target_network()

        return max_q

class ActorNetwork():
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -5 and 5
    """
    def __init__(self, sess, state_dim, action_dim, action_bounds, learning_rate, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.lr = learning_rate
        self.tau = tau

        self.inputs, self.out, self.scaled_out = self.create_actor_network('AN')
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network('target_AN')

        self.t_an_params = tf.get_collection('target_AN_params')
        self.an_params = tf.get_collection('AN_params')
        
        self.update_target_network_params = \
            [tf.assign(t, tf.multiply(e, self.tau) + tf.multiply(t, 1. - self.tau)) \
            for t, e in zip(self.t_an_params, self.an_params)]
        
        self.action_grad = tf.placeholder(tf.float32, [None, self.action_dim], name="action_grad")
        
        self.actor_grads = tf.gradients(self.scaled_out, self.an_params, -self.action_grad)

        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_grads, self.an_params))

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def create_actor_network(self, net_name):
        """
        Creates the Actor Network in TensorFlow
        """
        c_name = [net_name + '_params', tf.GraphKeys.GLOBAL_VARIABLES]

        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim], name="actor_inputs")
        layer1_units = 400
        layer2_units = 300

        with tf.name_scope(net_name):
            with tf.name_scope('layer1'):
                weights = tf.Variable(tf.truncated_normal([self.state_dim, layer1_units],
                            mean=0.0, stddev=0.02), name='weights', collections=c_name)
                biases = tf.Variable(tf.zeros([layer1_units]), name='biases', collections=c_name)
                layer1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

            with tf.name_scope('layer2'):
                weights = tf.Variable(tf.truncated_normal([layer1_units, layer2_units],
                            mean=0.0, stddev=0.02), name='weights', collections=c_name)
                self.variable_summaries(weights)
                biases = tf.Variable(tf.zeros([layer2_units]), name='biases', collections=c_name)
                self.variable_summaries(biases)
                layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
            
            with tf.name_scope('tanh'):
                weights = tf.Variable(tf.random_uniform([layer2_units, self.action_dim],
                            -0.003, 0.003), name='weights', collections=c_name)
                self.variable_summaries(weights)
                biases = tf.Variable(tf.zeros([self.action_dim]), name='biases', collections=c_name)
                self.variable_summaries(biases)
                out = tf.nn.tanh(tf.matmul(layer2, weights) + biases)
                action_range = self.action_bounds[1] - self.action_bounds[0]
                action_offset = float(self.action_bounds[0] + self.action_bounds[1])/2.0
                scaled_out = out*action_range/2.0 + action_offset
        return inputs, out, scaled_out

    def train(self, inputs, action_grad):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_grad: action_grad
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
    
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class CriticNetwork():
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau): #, num_actor_vars):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.tau = tau

        self.inputs, self.action, self.out = self.create_critic_network('CN')
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network('target_CN')

        self.t_cn_params = tf.get_collection('target_CN_params')
        self.cn_params = tf.get_collection('CN_params')

        self.update_target_network_params = \
            [tf.assign(t, tf.multiply(e, self.tau) + tf.multiply(t, 1. - self.tau)) \
            for t, e in zip(self.t_cn_params, self.cn_params)]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name="predicted_q_value")

        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)
    
    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def create_critic_network(self, net_name):
        """
        Creates the Critic Network in TensorFlow
        """
        c_name = [net_name + '_params', tf.GraphKeys.GLOBAL_VARIABLES]
        inputs = tf.placeholder(tf.float32, [None, self.state_dim], name="critic_inputs")
        action = tf.placeholder(tf.float32, [None, self.action_dim], name="critic_actions")

        layer1_units = 400
        layer2_units = 300

        with tf.name_scope(net_name):
            with tf.name_scope('layer1'):
                weights = tf.Variable(tf.truncated_normal([self.state_dim, layer1_units],
                            mean=0.0, stddev=0.02), name='weights', collections=c_name)
                self.variable_summaries(weights)
                biases1 = tf.Variable(tf.zeros([layer1_units]), name='biases', 
                            collections=c_name)
                self.variable_summaries(biases1)
                pre_activation = tf.matmul(inputs, weights) + biases1
                layer1 = tf.nn.relu(pre_activation)

            # Add the action tensor into the second layer
            with tf.name_scope('layer2'):
                with tf.name_scope('with_inputs'):
                    i_weights = tf.Variable(tf.truncated_normal([layer1_units, layer2_units],
                                                mean=0.0, stddev=0.02), name='weights', 
                                                collections=c_name)
                    self.variable_summaries(i_weights)
                with tf.name_scope('with_action'):
                    a_weights = tf.Variable(tf.truncated_normal([self.action_dim, layer2_units],
                                            mean=0.0, stddev=0.02), name='weights', 
                                            collections=c_name)
                    self.variable_summaries(a_weights)
                biases2 = tf.Variable(tf.zeros([layer2_units]), name='biases', 
                                            collections=c_name)
                self.variable_summaries(biases2)

                layer2 = tf.nn.relu(tf.matmul(layer1,i_weights) + tf.matmul(action,a_weights) + biases2)
            
            with tf.name_scope('linear'):
                # linear layer connected to 1 output representing Q(s,a)
                # Weights are init to Uniform[-3e-3, 3e-3]
                weights_lin = tf.Variable(tf.random_uniform([layer2_units, 1],-0.003,0.003), collections=c_name)
                self.variable_summaries(weights_lin)
                biases_lin = tf.Variable(tf.zeros([1]), name='biases', collections=c_name)
                self.variable_summaries(biases_lin)
                out = tf.matmul(layer2,weights_lin) + biases_lin
            
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })
    
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })
    
    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

def main():
    with tf.Session() as sess:
        env = WorkSpace(preset='B', continuous=True)

        state_dim = env.observation_space
        action_dim = 2
        action_bounds = [-5., 5.]

        ddpg = DDPG(sess, state_dim, action_dim, action_bounds)
        ddpg.train(sess, env, resume=False, visualize=True)

if __name__ == '__main__':
    # tf.app.run()
    main()