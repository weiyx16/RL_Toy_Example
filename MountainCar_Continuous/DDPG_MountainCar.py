"""
Environment for MountainCarContinous-V0 from gym
Src: https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
ref: https://github.com/lirnli/OpenAI-gym-solutions
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

Requirement:
Tensorflow: 1.12.0
gym: 0.15.4
"""

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Actor(object):
    """
        Actor of DDPG
    """
    def __init__(self, sess, n_features, action_bound):
        self.sess = sess

        self.cur_s = tf.placeholder(tf.float32, [None, n_features], name="cur_state")
        self.next_s = tf.placeholder(tf.float32, [None, n_features], name="next_state")

        self.n_hidden1 = 40
        self.n_hidden2 = 40
        self.n_outputs = 1

        self.lr = 1e-4 #2e-5
        self.tau = 0.01

        with tf.variable_scope("Actor"):
            self.action = self._build_net('Eval', trainable=True)
            self.action_t = self._build_net('Target', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Eval')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Target')

        self.soft_replace_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, scope, trainable):
        with tf.variable_scope(scope):
            init_xavier = tf.contrib.layers.xavier_initializer()
            if trainable:
                hidden1 = tf.layers.dense(inputs=self.cur_s, units=self.n_hidden1, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l1",trainable=trainable)
            else:
                hidden1 = tf.layers.dense(inputs=self.next_s, units=self.n_hidden1, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l1",trainable=trainable)
            hidden2 = tf.layers.dense(inputs=hidden1, units=self.n_hidden2, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l2",trainable=trainable) 
            # Deterministic 确定性输出，所以没有miu和sigma来估计分布并采样的操作
            action = tf.layers.dense(inputs=hidden2, units=self.n_outputs, activation=None, kernel_initializer=init_xavier, name="Actor_action",trainable=trainable)
            action_output = tf.clip_by_value(action, action_bound[0], action_bound[1])
            # action_output = tf.multiply(actions, self.action_bound, name='scaled_a')
            return action_output
    
    def actor_grad(self):
        with tf.variable_scope("policy_grad"):
            self.action_grads = tf.placeholder(tf.float32,[None,self.n_outputs])
            self.policy_grads = tf.gradients(self.action,self.e_params,-action_grads)
            self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(policy_grads,self.e_params))

    def optimize(self, cur_s, action_grads):
        cur_s = np.reshape(cur_s,(1,2))
        self.sess.run(self.train_op, feed_dict={self.cur_s:cur_s, self.action_grads:action_grads})
        self.sess.run(self.soft_replace_op)
        
    def choose_action(self, cur_s):
        cur_s = np.reshape(cur_s,(1,2))
        return self.sess.run(self.action, {self.cur_s: cur_s})
    
    def evaluate_target_actor(self, next_s):
        next_s = np.reshape(next_s,(1,2))
        return self.sess.run(self.action_t, feed_dict={self.next_s:next_s})

# TODO: