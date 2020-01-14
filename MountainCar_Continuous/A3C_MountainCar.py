"""
Environment for MountainCarContinous-V0 from gym
Src: https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
ref: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A3C/
    https://github.com/stefanbo92/A3C-Continuous
    https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python
Requirement:
Tensorflow: 1.12.0
gym: 0.15.4
TODO:
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import os
import threading
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# HyperParameters:
## Environment
RENDER = False  # Show GUI
env = gym.make('MountainCarContinuous-v0')
# env.seed(1)     # reproducible
# env = env.unwrapped  # remove the 200 time step limit the cart pole example defaults to

print('-'*30+'\r\nAction Space: {}\r\n'.format(env.action_space)) # Box(1) Left:neg;Right:pos range:[-1,1]
print('-'*30+'\r\nObservation Space: {}\r\n'.format(env.observation_space)) 
state_range = [env.low_state, env.high_state]
action_bound = [env.action_space.low, env.action_space.high]
n_features_state = env.observation_space.shape[0]
n_features_action = env.action_space.shape[0]

## Training
num_workers = multiprocessing.cpu_count()
epoch_global = 50 * num_workers
print('-'*30+'\r\nTraining A3C with worker: {}\r\n'.format(num_workers))
global_scope_name = 'Global_Net'
# sets how often the global network should be updated
update_global = 10
gamma = 0.99
# entropy factor
entropy_beta = 0.01 
lr_actor = 1e-4
lr_critic = 1e-3
global_rewards = []
global_episodes = 0

class A3C(object):
    """
        Global A3C model including actor and critic in the same time
    """
    def __init__(self, scope_name, sess, globalAC=None):
        self.sess=sess
        
        self.actor_optimizer = tf.train.RMSPropOptimizer(lr_actor, name='Actor_Optim')
        self.critic_optimizer = tf.train.RMSPropOptimizer(lr_critic, name='Critic_Optim')

        # if the network is the global one
        if scope_name == global_scope_name:
            with tf.variable_scope(scope_name):
                # initialize states and build actor and critic network
                self.cur_s = tf.placeholder(tf.float32, [None, n_features_state], 'cur_state')
                # get the parameters of actor and critic networks
                _, _, _, self.a_params, self.c_params = self._build_net(scope_name)
        else:
            with tf.variable_scope(scope_name):
                self.cur_s = tf.placeholder(tf.float32, [None, n_features_state], 'cur_state')
                self.a_in = tf.placeholder(tf.float32, [None, n_features_action], 'action_in')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'target_v')

                mu, sigma, self.state2value, self.a_params, self.c_params = self._build_net(scope_name)

                # then we calculate td error as the difference between v_target - v
                td_error = tf.subtract(self.v_target, self.state2value, name='td_error')

                with tf.name_scope('Loss_Critic'):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                with tf.name_scope('wrap_action'):
                    mu = mu * action_bound[1]
                    sigma = sigma + 1e-5                   
                
                # we can generate distribution using this updated mean and var
                norm_dist = tf.distributions.Normal(mu, sigma)
    
                with tf.name_scope('Loss_Actor'):
                    self.actor_loss = norm_dist.log_prob(self.a_in) * td_error 
                    self.actor_loss += entropy_beta * norm_dist.entropy()
                    self.actor_loss = tf.reduce_mean(-self.actor_loss)
                    
                with tf.name_scope('choose_action'):
                    self.action_sampled = tf.clip_by_value(norm_dist.sample(1), action_bound[0], action_bound[1])
     
                # calculate gradients to assign to different works
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.actor_loss, self.a_params)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_params)

            with tf.name_scope('sync'):
                assert globalAC is not None, "Can't find global A-C network for paramters assignment"
                with tf.name_scope('pull'):
                    # assign local params with global params
                    self.pull_a_params_op = [local_param.assign(global_param) for local_param, global_param in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [local_param.assign(global_param) for local_param, global_param in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    # update global params with grad from local params
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope_name):
        """
            Main Struture of Actor-Critic
        """
        actor_n_hidden1 = 64
        actor_n_hidden2 = 128
        critic_n_hidden1 = 32
        critic_n_hidden2 = 64
        init_xavier = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Actor"):
            hidden1_actor = tf.layers.dense(inputs=self.cur_s, units=actor_n_hidden1, activation=tf.nn.relu6, kernel_initializer=init_xavier, name="Actor_l1")
            hidden2_actor = tf.layers.dense(inputs=hidden1_actor, units=actor_n_hidden2, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Actor_l2") 
            mu = tf.layers.dense(inputs=hidden2_actor, units=n_features_action, activation=tf.nn.tanh, kernel_initializer=init_xavier, name="Actor_mu")  #tf.nn.tanh
            sigma = tf.layers.dense(inputs=hidden2_actor, units=n_features_action, activation=tf.nn.softplus, kernel_initializer=init_xavier, name="Actor_sigma")
        with tf.variable_scope("Critic"):
            hidden1_critic = tf.layers.dense(inputs=self.cur_s, units=critic_n_hidden1, activation=tf.nn.relu6, kernel_initializer=init_xavier, name="Critic_l1")
            hidden2_critic = tf.layers.dense(inputs=hidden1_critic, units=critic_n_hidden2, activation=tf.nn.elu, kernel_initializer=init_xavier, name="Critic_l2") 
            state2value = tf.layers.dense(inputs=hidden2_critic, units=1, activation=None, kernel_initializer=init_xavier, name="Critic_V")
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name + '/Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name + '/Critic')
        return mu, sigma, state2value, a_params, c_params
    
    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def assign_local(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, cur_s):
        cur_s = np.reshape(cur_s,(1,2))
        return self.sess.run(self.action_sampled, {self.cur_s: cur_s})

class Worker(object):
    def __init__(self, local_scope_name, sess, globalAC):
        # intialize environment for each worker
        self.env = gym.make('MountainCarContinuous-v0')#.unwrapped 
        self.local_scope_name = local_scope_name
        
        # create A3C agent for each worker
        assert globalAC is not None, "Can't find global A-C network for paramters assignment"
        self.AC = A3C(local_scope_name, sess, globalAC)
        self.sess = sess
    
    def state_normalize(self, state):
        state = np.reshape(state,(1,2))
        low_s, high_s = state_range
        var_i = np.power((high_s - low_s), 2) / 12
        mean_i = (high_s + low_s) / 2
        state = (state - mean_i) / var_i
        return state

    def work(self):
        global global_rewards, global_episodes
        worker_iter = 1
 
        # store state, action, reward
        buffer_s, buffer_a, buffer_r = [], [], []
        
        # loop if the coordinator is active and global episode is less than the maximum episode
        while not coord.should_stop() and global_episodes < epoch_global:
            
            global_episodes += 1
            cur_state = self.state_normalize(self.env.reset())
            action_count, sum_reward = 0, 0
            done = False
            
            while not done:
                # Render the environment for only worker 1
                if self.local_scope_name == 'W_0' and RENDER: self.env.render()
                action_count += 1 
                cur_action = self.AC.choose_action(cur_state)
                next_state, reward, done, _ = self.env.step(cur_action)

                next_state = self.state_normalize(next_state)

                sum_reward += reward
                # store the state, action and rewards in the buffer
                buffer_s.append(cur_state)
                buffer_a.append(np.reshape(cur_action,(1,)))
                # normalize the reward
                buffer_r.append(reward) # (reward+8)/8
    
                # we Update the global network after particular time step
                if worker_iter % update_global == 0 or done:
                    if done:
                        next_v = 0
                    else:
                        next_v = self.sess.run(self.AC.state2value, {self.AC.cur_s: next_state})[0, 0]
                    
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        # Gt = r + gamma*(Gt+1)
                        # 最后一个Gt就是cur_state对应的。
                        next_v = r + gamma * next_v
                        buffer_v_target.append(next_v)    
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    # update global network
                    self.AC.update_global({self.AC.cur_s: buffer_s,
                                            self.AC.a_in: buffer_a,
                                            self.AC.v_target: buffer_v_target,})
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.assign_local()
                    
                cur_state = next_state
                worker_iter += 1
            
            global_rewards.append(sum_reward)
            print(" >> episode:", global_episodes, " average reward to now:",np.mean(global_rewards))
            if np.mean(global_rewards[-100:]) > 90 and len(global_rewards) >= 101:
                print("****************Solved***************")
                print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(np.mean(global_rewards[-100:])))
        
        self.env.close()

def state_rescale(state, state_range):
    low_s, high_s = state_range
    for state_i in range(len(low_s)):
        state[state_i] = (state[state_i] - low_s[state_i]) / (high_s[state_i] - low_s[state_i])
    return state

if __name__ == "__main__":

    sess = tf.Session()
    with tf.device("/cpu:0"):
        # global model    
        global_ac = A3C(global_scope_name, sess, None)

        # local model
        workers = []
        for i in range(num_workers):
            local_scope_name = 'W_%i' % i
            workers.append(Worker(local_scope_name, sess, global_ac))
    
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
   
    #start workers
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

    plt.plot(global_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('./img/A3C.png')
    plt.close()