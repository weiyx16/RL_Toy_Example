"""
Environment for MountainCar-V0 from gym
Src: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
Ref: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

Requirement:
Tensorflow: 1.12.0
gym: 0.15.4
TODO:
"""
import gym
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
import matplotlib.pyplot as plt

class DeepQNetwork:
    def __init__(
            self,
            sess,
            n_actions,
            n_features
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.tau = 0.01
        self.memory_size = 100000
        self.batch_size = 64
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999

        # initialize zero memory [s, a, r, s_, done]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        self.soft_replace_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
                                 for t, e in zip(self.t_params, self.e_params)]
        self.hard_replace_op = [tf.assign(t, e)
                                 for t, e in zip(self.t_params, self.e_params)]
    def _build_net(self):
        n_hidden1 = 32
        n_hidden2 = 64
        n_hidden3 = 32
        # ------------------ all inputs ------------------------
        self.cur_s = tf.placeholder(tf.float32, [None, self.n_features], name='cur_state')  # input State
        self.next_s = tf.placeholder(tf.float32, [None, self.n_features], name='next_state')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.dones = tf.placeholder(tf.float32, [None, ], name='done')  # termination

        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        init_xavier = tf.contrib.layers.xavier_initializer()
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            hidden1_eval = tf.layers.dense(inputs=self.cur_s, units=n_hidden1, activation=tf.nn.relu, kernel_initializer=init_xavier, name="Eval_l1")
            hidden2_eval = tf.layers.dense(inputs=hidden1_eval, units=n_hidden2, activation=tf.nn.relu, kernel_initializer=init_xavier, name="Eval_l2") 
            hidden3_eval = tf.layers.dense(inputs=hidden2_eval, units=n_hidden3, activation=tf.nn.relu, kernel_initializer=init_xavier, name="Eval_l3")
            self.q_eval = tf.layers.dense(inputs=hidden3_eval, units=self.n_actions, activation=None, kernel_initializer=init_xavier, name="Eval_q")

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            hidden1_tgt = tf.layers.dense(inputs=self.next_s, units=n_hidden1, activation=tf.nn.relu, name="Tgt_l1")
            hidden2_tgt = tf.layers.dense(inputs=hidden1_tgt, units=n_hidden2, activation=tf.nn.relu, name="Tgt_l2")
            hidden3_tgt = tf.layers.dense(inputs=hidden2_tgt, units=n_hidden3, activation=tf.nn.relu, name="Tgt_l3") 
            self.q_next = tf.layers.dense(inputs=hidden3_tgt, units=self.n_actions, activation=None, name="Tgt_q")

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')*(1-self.dones)    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, dones):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # s = np.reshape(s,(2,))
        
        transition = np.hstack((s, [a, r], s_, dones))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_t_q(self):
        self.sess.run(self.hard_replace_op)

    def choose_action(self, observation, epsilon_decay=False):
        if epsilon_decay:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.cur_s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def optimize(self):

        self.sess.run(self.soft_replace_op)
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.cur_s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.next_s: batch_memory[:, -self.n_features-1:-1],
                self.dones: batch_memory[:, -1],
            })

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return cost

def state_normalize(state, state_range):
    state = np.reshape(state,(2,))
    low_s, high_s = state_range
    var_i = np.power((high_s - low_s), 2) / 12
    mean_i = (high_s + low_s) / 2
    state = (state - mean_i) / var_i
    return state

def reward_warp(state):
    if state[0] >= 0.5:
        return 10
    # if state[0] > -0.4:
    #     return (1+state[0])**2
    return -1

if __name__ == "__main__":
    ## Environment
    RENDER = False  # Show GUI
    env = gym.make('MountainCar-v0')
    # env.seed(1)     # reproducible
    # env = env.unwrapped  # remove the 200 time step limit the cart pole example defaults to

    print('-'*30+'\r\nAction Space: {}\r\n'.format(env.action_space)) # Box(1) Left:neg;Right:pos range:[-1,1]
    print('-'*30+'\r\nObservation Space: {}\r\n'.format(env.observation_space)) 
    state_range = [env.low, env.high]
    # self.state = (position, velocity) range:[-1.2,-0.07]~[0.6,0.07] Box(2)

    ## Agent
    sess = tf.Session()
    agent = DeepQNetwork(sess, n_features=2, n_actions=3)
    sess.run(tf.global_variables_initializer())
    agent.update_t_q()

    ## Training
    epoch = 1000
    warmup_iter = 500
    iter_over_epoch = 0
    episode_rewards = []
    episode_attitude = []
    for iepoch in tqdm(range(epoch), ncols=70):

        cur_state = state_normalize(env.reset(), state_range)
        action_count, sum_reward, iepoch_max_atti = 0, 0.0, env.observation_space.low[0]
        loss_epoch = 0
        done = False
        
        while not done:
            if RENDER: env.render()
            cur_action = agent.choose_action(cur_state)
            next_state, reward, done, _ = env.step(cur_action)  # in step: reward = 0 - math.pow(action[0],2)*0.1
            
            iepoch_max_atti = max(iepoch_max_atti, next_state[0])
            action_count += 1
            sum_reward += reward

            reward = reward_warp(next_state)
            next_state = state_normalize(next_state, state_range)
            agent.store_transition(cur_state, cur_action, reward, next_state, float(done))

            if iter_over_epoch < warmup_iter:
                iter_over_epoch += 1
            else:
                loss = agent.optimize()
                loss_epoch += loss

            cur_state = next_state
        
        episode_rewards.append(sum_reward)
        episode_attitude.append(iepoch_max_atti)  #这是随着时间变化，car可以到达的最高attitude
        print("\r\n >> episode:", iepoch, " action tried:", int(action_count), " Heighest:",iepoch_max_atti, " Loss:",loss_epoch/action_count)
        if np.mean(episode_rewards[-100:]) > -110 and len(episode_rewards) >= 101:
            print("****************Solved***************")
            print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(np.mean(episode_rewards[-100:])))

    env.close()
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('./img/DQN_reward_warp_200.png')
    plt.close()