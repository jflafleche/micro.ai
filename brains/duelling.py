"""
This code implements a Double Duelling DQN network.
Based on the paper by Wang et al.
    https://arxiv.org/pdf/1511.06581.pdf

The code was implemented with the Tensorflow framework,
with help from the insightful post by Arthur Juliani:
    https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df

Author: Jean-Francois Lafleche, 2017
"""

import numpy as np
import tensorflow as tf

class DuellingDQN():
    def __init__(
        self, 
        sess, 
        env_params,
        brain_params
    ):
        self.sess = sess
        self.state_dim = env_params['state_dim']
        self.action_dim = env_params['action_dim']
        self.lr = brain_params['lr_1']
        self.discount_factor = brain_params['gamma']
        self.tau = brain_params['tau'] # soft target update param

        self.inputs, self.q_out = self.create_network(['eval', tf.GraphKeys.GLOBAL_VARIABLES])
        self.t_inputs, self.t_q_out = self.create_network(['target', tf.GraphKeys.GLOBAL_VARIABLES])

        self.t_params = tf.get_collection('target')
        self.e_params = tf.get_collection('eval')

        self.update_target_network_params = \
            [tf.assign(t, tf.multiply(e, self.tau) + tf.multiply(t, 1. - self.tau)) \
            for t, e in zip(self.t_params, self.e_params)]

        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.action_dim, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

        self.predicted_q_value = tf.placeholder(tf.float32, [None], name="predicted_q_value")

        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.Q)

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def create_network(self, c_name):
        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='inputs')
        layer1_units = 100
        layer2_units = 50

        with tf.name_scope('layer1'):
            w1 = tf.Variable(tf.truncated_normal([self.state_dim, layer1_units],
                            mean=0.0, stddev=0.02), name='weights1', collections=c_name)
            b1 = tf.Variable(tf.zeros([layer1_units]), name='biases1', collections=c_name)
            layer1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
        
        with tf.name_scope('layer2'):
            w2 = tf.Variable(tf.truncated_normal([layer1_units, layer2_units],
                            mean=0.0, stddev=0.02), name='weights2', collections=c_name)
            b2 = tf.Variable(tf.zeros([layer2_units]), name='biases2', collections=c_name)
            layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

        with tf.name_scope('value'):
            w_value = tf.Variable(tf.truncated_normal([layer2_units, 1],
                            mean=0.0, stddev=0.02), name='weights_val', collections=c_name)
            b_value = tf.Variable(tf.zeros([1]), name='biases_val', collections=c_name)
            value = tf.matmul(layer2, w_value) + b_value
        
        with tf.name_scope('advantage'):
            w_adv = tf.Variable(tf.truncated_normal([layer2_units, self.action_dim],
                            mean=0.0, stddev=0.02), name='weights_adv', collections=c_name)
            b_adv = tf.Variable(tf.zeros([self.action_dim]), name='biases_adv', collections=c_name)
            advantage = tf.matmul(layer2, w_adv) + b_adv
        
        # Eqn. (9) from https://arxiv.org/pdf/1511.06581.pdf
        q_out = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)

        return inputs, q_out

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
        target_q = self.predict_target(states_)

        y = rewards + self.discount_factor*target_q[np.arange(batch_size),actions]*(-stops)

        q_out, _ = self.train(states, actions, y)

        return np.amax(q_out)

    def update(self):
        self.sess.run(self.update_target_network_params)

    def train(self, inputs, action, predicted_q):
        asf = self.sess.run([self.q_out, self.optimize, self.loss, self.Q], feed_dict={
            self.inputs: inputs,
            self.actions: action,
            self.predicted_q_value: predicted_q
        })
        # print(asf[2])

        return asf[:2]

    def predict(self, state):
        state = np.reshape(state, (-1, self.state_dim))
        q_out = self.sess.run(self.q_out, feed_dict={self.inputs: state})
        return np.argmax(q_out)

    def predict_batch(self, state):
        state = np.reshape(state, (-1, self.state_dim))
        q_out = self.sess.run(self.q_out, feed_dict={self.inputs: state})

        return np.argmax(q_out,1)

    def predict_target(self, inputs):
        return self.sess.run(self.t_q_out, feed_dict={
            self.t_inputs: inputs
        })