"""
Sarsa Lambda Agent
"""

import numpy as np
from copy import deepcopy

class Sarsalambda(object):
    def __init__(self, actions, states=None, learning_rate=0.01, gamma=0.9, slambda=0.9, epsilon=0.9, begin_epsilon=0.7):
        self.actions = actions
        self.states = states
        self.lr = learning_rate
        self.gamma = gamma
        self.slambda = slambda
        self.epsilon = epsilon
        self.begin_epsilon = begin_epsilon
        # Initialize Q table
        self.Q = np.random.uniform(low = -1, high = 1, 
                            size = (int(states[0]), int(states[1]), len(actions)))
        self.E = deepcopy(self.Q)

    def choose_action(self, state, process):
        # action selection
        if np.random.uniform() < (self.begin_epsilon + (self.epsilon-self.begin_epsilon) * process):
            # choose best action
            action = np.argmax(self.Q[state[0],state[1], :])
        else:
            # choose random action
            action = np.random.randint(0, len(self.actions))
        return action

    def learn(self, cur_s, cur_a, r, next_s, next_a):
        q_predict = self.Q[cur_s[0], cur_s[1], cur_a]
        if next_s is not 'done':
            delta = r + self.gamma * self.Q[next_s[0],next_s[1], next_a] - q_predict
            self.E[cur_s[0], cur_s[1], cur_a] = 1
            self.Q += self.lr*delta*self.E
            self.E = self.gamma*self.slambda*self.E
        else:
            delta = r - q_predict
            self.E[cur_s[0], cur_s[1], cur_a] = 1
            self.Q += self.lr*delta*self.E
            self.E = np.zeros_like(self.E)
        # Method 1: 
        # self.E[cur_s[0], cur_s[1], cur_a] = self.E[cur_s[0], cur_s[1], cur_a] + 1
        # Method 2:
        # self.E[cur_s[0], cur_s[1], :] *= 0
        # self.E[cur_s[0], cur_s[1], cur_a] = 1
        # self.Q += self.lr*delta*self.E
        # self.E = self.gamma*self.slambda*self.E