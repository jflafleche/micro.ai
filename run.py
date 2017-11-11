import datetime
import argparse
import pprint as pp
import time

import numpy as np
import tensorflow as tf
import random

import gym
from custom_env.custom_envs import CustomEnvs

from brains.brains import Brains
from memories.memories import Memories
from agent import Agent

import keyboard

def train(
    sess, 
    env,
    env_params,
    agent,
    agent_params,
    max_episodes,
    max_episode_len,
    resume, 
    render
    ):
    """
    Trains the agent to obtain maximum reward in the environment.

    Arguments:
        sess: TensorFlow session
        env: environment (default from openai gym)
        agent: agent initialized with memory and brain
        max_episodes: maximum number of episodes for training
        max_episode_len: maximum length of a single episode
        resume: bool; if True, resume from last checkpoint
        render: bool; if True, render environment
    """
    path = env_params['env']
    path = path + '/trained_agent' if agent_params['trained_agent'] else path
    SUMMARY_DIR = './logs/' + path + '/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINT_DIR = './checkpoints/' + path

    state_dim = env_params['state_dim']
    action_dim = env_params['action_dim']
    action_bounds = env_params['action_bounds']
    r_successes = 0
    r_rewards = 0
    ep_avg_max_q = 0
    
    # VISUALIZE
    if render:
        env.render()

    summary_vars, summary_ops = build_summaries()

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    saver = tf.train.Saver()

    # agent.update()

    # LOADER
    if resume or agent_params['trained_agent']:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        saver.restore(sess,ckpt.model_checkpoint_path)
    
    for episode in range(max_episodes):
        success = 0
        ep_reward = 0
        ep_avg_max_q = 0

        s = env.reset()
        
        # SAVER
        if (episode % 100 == 0 and episode > 1):
            print('Saving Model %i ...' % episode)
            save(sess, saver, CHECKPOINT_DIR, episode)
        
        for step in range(1, max_episode_len):
            a = agent.act(s)

            s_, r, done, info = env.step(a)

            ep_reward += r
            if info == 'success':
                success += 1

            s = s.reshape(state_dim,)
            a = a.reshape(action_dim,)
            s_ = s_.reshape(state_dim,)
            
            experience = (s, a, r, s_, done)
            agent.observe(experience)
            max_q = agent.replay()
            agent.update()
            ep_avg_max_q += max_q

            s = s_.copy()
            critic_loss = 0
            if done or step == max_episode_len-1:
                ep_avg_max_q /= step
                r_rewards, r_successes = print_to_console(episode, r_rewards, r_successes, ep_avg_max_q, ep_reward, success)
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_avg_max_q,
                    summary_vars[2]: success
                })
                summary_writer.add_summary(summary_str, episode)
                break

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_avg_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_avg_max_q)
    success = tf.Variable(0.)
    tf.summary.scalar("Goals", success)

    summary_vars = [episode_reward, episode_avg_max_q, success]
    summary_ops = tf.summary.merge_all()

    return summary_vars, summary_ops

def save(sess, saver, path, episode):
    saver.save(sess, path+'/model-'+str(episode)+'.cptk')

def print_to_console(episode, running_rewards, running_successes, ep_avg_max_q, ep_reward, success):
    if episode == 0:
        running_successes = success
        running_rewards = ep_reward
    else:
        running_successes = running_successes*0.99 + success*0.01
        running_rewards = running_rewards*0.99 + ep_reward*0.01

    # writer.flush()
    row_format = "{:<15}" * 6
    print(row_format.format(
        # '1', '2','3','4','5','6'
        'Episode {:d}'.format(episode),
        '| maxQ: {:.4f}'.format(ep_avg_max_q),
        '| Reward: {:d}'.format(int(ep_reward)), 
        '| Goals: {:d}'.format(success),
        '| Goal Avg: {:.2f}'.format(running_successes),
        '| Reward Avg: {:.2f}'.format(running_rewards)
    ))
    return running_rewards, running_successes

def manual_control(env):
    env.render()
    env.reset()

    success = 0
    ep_reward = 0
    episode = 1
    raw_input('ready?')
    for step in range(1000):
        a = np.zeros(2)

        if keyboard.is_pressed('right'):
            a[0] = 5.0
        elif keyboard.is_pressed('left'):
            a[0] = -5.0
        
        if keyboard.is_pressed('up'):
            a[1] = 5.0
        elif keyboard.is_pressed('down'):
            a[1] = -5.0
        
        time.sleep(0.05)

        s_, r, done, info = env.step(a)
        time.sleep(0.05)

        ep_reward += r
        if r == 1000:
            success += 1

        if done:
            r_rewards = 0
            r_successes = 0
            ep_avg_max_q = 0
            print_to_console(episode, r_rewards, r_successes, ep_avg_max_q, ep_reward, success)
            break

def main(args):
    with tf.Session() as sess:
        tf.set_random_seed(int(args['random_seed']))

        # ENV SETUP
        if 'microbot' in args['env']:
            custom_envs = CustomEnvs()
            env = custom_envs.envs[args['env']]
        else:
            env = gym.make(args['env'])
            env.seed(int(args['random_seed']))
            # determine type of action space
        
        action_space_type = env.action_space.__class__.__name__

        if action_space_type == 'Discrete':
            action_dim = env.action_space.n
            action_bounds = None
        else:
            action_dim = env.action_space.shape[0]
            action_bounds = [env.action_space.low[0], env.action_space.high[0]]

        env_params = {
            'env': args['env'],
            'env_type': args['env_type'],
            'state_dim': env.observation_space.shape[0],
            'action_dim': action_dim,
            'action_bounds': action_bounds
        }

        # BRAIN SETUP
        brain_params = {
            'lr_1': float(args['lr_1']),
            'lr_2': float(args['lr_2']),
            'tau': float(args['tau']),
            'gamma': float(args['gamma'])
        }
        brains = Brains(
            sess,
            env_params,
            brain_params
        )
        brain = brains.get(args['brain'])

        # MEMORY SETUP
        memory_params = {
            'capacity': int(args['memory_capacity']),
            'minibatch_size': int(args['minibatch_size'])
        }
        memories = Memories(memory_params)
        memory = memories.get(args['memory'])

        # AGENT SETUP
        agent_params = {
            'trained_agent': args['trained_agent'],
            'exploration_type': args['exploration_type'],
            'exploration_init': float(args['exploration_init']),
            'exploration_final': float(args['exploration_final']),
            'exploration_rate': float(args['exploration_rate'])
        }
        agent = Agent(memory, brain, env_params, agent_params)

        if args['manual']:
            manual_control(env)
        else:
            train(
                sess,
                env,
                env_params,
                agent,
                agent_params,
                args['max_episodes'], 
                args['max_episode_len'],
                args['resume'],
                args['render_env']
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    # brain parameters
    parser.add_argument('--brain', help='brain choice', default='ddpg-tf')
    parser.add_argument('--lr-1', help='general or actor network learning rate', default=0.0001)
    parser.add_argument('--lr-2', help='critic network learning rate', default=0.001)

    # memory parameter
    parser.add_argument('--memory', help='memory choice', default='experience-replay')
    parser.add_argument('--memory-capacity', help='max capacity of the memory buffer', default=1000000)

    # agent parameters
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument('--trained-agent', help='load a trained agent', action='store_true')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--env-type', help='specify either a discrete or continuous action space', default='noise')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=10000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--resume', help='resume from last checkpoint', action='store_true')
    parser.add_argument('--exploration-type', help='choose between exploring with "random" action or "noise"', default='noise')
    parser.add_argument('--exploration-init', help='initial chance of random action', default=0.9)
    parser.add_argument('--exploration-final', help='final chance of random action', default=0.1)
    parser.add_argument('--exploration-rate', help='decay rate of chance of random action', default=0.005)
    parser.add_argument('--manual', help='manual control of agent', action='store_true')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)