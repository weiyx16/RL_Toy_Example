"""
Environment for MountainCarContinous-V0 from gym
Src: https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
ref: https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
Requirement:
Tensorflow: 1.12.0
gym: 0.15.4
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


RENDER = False  # Show GUI


class Actor(object):
    """
        Policy part to sample action
    """
    def __init__(self, sess, n_features, action_bound):
        self.sess = sess

        self.cur_s = tf.placeholder(tf.float32, [1, n_features], name="cur_state")
        self.a_in = tf.placeholder(tf.float32, None, name="action_in")
        self.TD_error = tf.placeholder(tf.float32, None, name="td_error")

        self.n_hidden1 = 40
        self.n_hidden2 = 40
        self.n_outputs = 1

        self.lr = 2e-5

        with tf.variable_scope("Actor"):
            init_xavier = tf.contrib.layers.xavier_initializer()
            
            hidden1 = tf.layers.dense(inputs=self.cur_s, units=self.n_hidden1, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l1")
            hidden2 = tf.layers.dense(inputs=hidden1, units=self.n_hidden2, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l2") 
            self.mu = tf.layers.dense(inputs=hidden2, units=self.n_outputs, activation=None, kernel_initializer=init_xavier, name="Actor_mu")  #tf.nn.tanh
            self.sigma = tf.layers.dense(inputs=hidden2, units=self.n_outputs, activation=tf.nn.softplus, kernel_initializer=init_xavier, name="Actor_sigma")
            self.sigma = self.sigma + 1e-5
            # 给定policy的估计之后，构建对应统计量的正正态分布，动作就是这个分布的采样
            self.norm_dist = tf.distributions.Normal(self.mu, self.sigma)
            self.action_sampled = tf.clip_by_value(self.norm_dist.sample(1), 
                                            action_bound[0], 
                                            action_bound[1])
        with tf.variable_scope("Loss_Actor"):
            self.loss = -tf.log(self.norm_dist.prob(self.a_in) + 1e-5) * self.TD_error
            # or
            # self.loss = -self.norm_dist.log_prob(self.a_in) * self.TD_error
            # self.loss += 0.001*self.norm_dist.entropy()
        
        with tf.variable_scope("Train_Actor"):
            self.train_op = tf.train.AdamOptimizer(self.lr, name="Actor_Optim").minimize(self.loss)
    
    def optimize(self, action, td_error, cur_s):
        cur_s = np.reshape(cur_s,(1,2))
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.a_in:action, self.TD_error:td_error, self.cur_s:cur_s})
        return loss
        
    def choose_action(self, cur_s):
        cur_s = np.reshape(cur_s,(1,2))
        # cur_s = cur_s[np.newaxis, :]  # tf.expand_dims
        return self.sess.run(self.action_sampled, {self.cur_s: cur_s})

class Critic(object):
    """
    Value function to return value of a certain state
    """
    def __init__(self, sess, n_features):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.next_v = tf.placeholder(tf.float32, [1, 1], name="next_v")
        self.r = tf.placeholder(tf.float32, name='reward')

        self.n_hidden1 = 400  
        self.n_hidden2 = 400
        self.n_outputs = 1
        
        self.lr = 1e-3
        self.gamma = 0.99

        # Structure
        with tf.variable_scope("Critic"):
            init_xavier = tf.contrib.layers.xavier_initializer()
            
            hidden1 = tf.layers.dense(inputs=self.s, units=self.n_hidden1, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Critic_l1")
            hidden2 = tf.layers.dense(inputs=hidden1, units=self.n_hidden2, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Critic_l2") 
            # self.state2value: value correspondent to self.s V_{\pi}(S_{t})
            self.state2value = tf.layers.dense(inputs=hidden2, units=self.n_outputs, activation=None, kernel_initializer=init_xavier, name="Critic_V")
        
        # Loss and training
        with tf.variable_scope("Loss_Critic"):
            TD_target = self.r + self.gamma * np.squeeze(self.next_v)
            self.TD_error = TD_target - self.state2value
            self.loss = tf.reduce_mean(tf.square(self.TD_error))

        with tf.variable_scope("Train_Critic"):
            self.train_op = tf.train.AdamOptimizer(self.lr, name="Critic_Optim").minimize(self.loss)
    
    def optimize(self, cur_s, next_s, r):
        # input S_{t+1} -> V_{\pi}(S_{t+1})
        # 注意在一次TD里，用同一拨参数去算next value 和 cur value。
        next_s = np.reshape(next_s,(1,2))
        cur_s = np.reshape(cur_s,(1,2))
        next_v = self.sess.run(self.state2value, {self.s: next_s})
        td_error, loss, _ = self.sess.run([self.TD_error, self.loss, self.train_op], feed_dict={self.r:r, self.next_v:next_v, self.s:cur_s})
        return td_error, loss

def state_rescale(state, state_range):
    low_s, high_s = state_range
    for state_i in range(len(low_s)):
        state[state_i] = (state[state_i] - low_s[state_i]) / (high_s[state_i] - low_s[state_i])
    return state

def state_normalize(state, state_range):
    state = np.reshape(state,(1,2))
    low_s, high_s = state_range
    var_i = np.power((high_s - low_s), 2) / 12
    mean_i = (high_s + low_s) / 2
    state = (state - mean_i) / var_i
    return state

if __name__ == "__main__":
    ## Environment
    env = gym.make('MountainCarContinuous-v0')
    # env.seed(1)     # reproducible
    # env = env.unwrapped  # remove the 200 time step limit the cart pole example defaults to

    print('-'*30+'\r\nAction Space: {}\r\n'.format(env.action_space)) # Box(1) Left:neg;Right:pos range:[-1,1]
    print('-'*30+'\r\nObservation Space: {}\r\n'.format(env.observation_space)) 
    state_range = [env.low_state, env.high_state]
    # self.state = (position, velocity) range:[-1.2,-0.07]~[0.6,0.07] Box(2)

    ## Agent
    sess = tf.Session()
    actor = Actor(sess, n_features=env.observation_space.shape[0], 
                        action_bound=[env.action_space.low,env.action_space.high])
    critic = Critic(sess, n_features=env.observation_space.shape[0])
    sess.run(tf.global_variables_initializer())

    ## Training
    epoch = 300
    episode_rewards = []
    episode_attitude = []
    for iepoch in range(epoch):

        cur_state = state_normalize(env.reset(), state_range)
        action_count, sum_reward, iepoch_max_atti = 0, 0.0, env.observation_space.low[0]
        critic_loss_all, actor_loss_all = 0,0
        done = False
        
        while not done:
            if RENDER: env.render()
            cur_action = actor.choose_action(cur_state)
            next_state, reward, done, _ = env.step(cur_action)  # in step: reward = 0 - math.pow(action[0],2)*0.1
            
            iepoch_max_atti = max(iepoch_max_atti, next_state[0])
            action_count += 1
            sum_reward += reward

            next_state = state_normalize(next_state, state_range)
            td_error, critic_loss = critic.optimize(cur_state, next_state, reward)
            actor_loss = actor.optimize(cur_action, td_error, cur_state)  
            # td_error or critic_loss 为什么算法写着导数才是td_error但是大家都不是那样实现的？
            # 但反正critic loss就是不 work，然后td error则最后loss都飞了？？？
            # 强行 abs(td_error) 或者先套个relu都不 work？？？

            cur_state = next_state
        
            critic_loss_all += critic_loss
            actor_loss_all += actor_loss
        
        episode_rewards.append(sum_reward)
        episode_attitude.append(iepoch_max_atti)  #这是随着时间变化，car可以到达的最高attitude
        print("\r\n >> {}, {}".format(critic_loss_all/action_count, actor_loss_all/action_count))
        print(" >> episode:", iepoch, " action tried:", int(action_count), " Heighest:",iepoch_max_atti, " average reward to now:",np.mean(episode_rewards))
        if np.mean(episode_rewards[-100:]) > 90 and len(episode_rewards) >= 101:
            print("****************Solved***************")
            print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(np.mean(episode_rewards[-100:])))

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('./img/A2C.png')
    plt.close()