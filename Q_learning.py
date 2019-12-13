"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np

class QLearningTable:
    def __init__(self, actions, states=None, learning_rate=0.01, gamma=0.9, epsilon=0.9, begin_epsilon=0.7):
        self.actions = actions
        self.states = states
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.begin_epsilon = begin_epsilon
        # Initialize Q table
        self.Q = np.random.uniform(low = -1, high = 1, 
                            size = (int(states[0]), int(states[1]), len(actions)))

    def choose_action(self, state, process):
        # action selection
        if np.random.uniform() < (self.begin_epsilon + (self.epsilon-self.begin_epsilon) * process):
            # choose best action
            action = np.argmax(self.Q[state[0],state[1], :])
        else:
            # choose random action
            action = np.random.randint(0, len(self.actions))
        return action

    def learn(self, cur_s, a, r, next_s):
        q_predict = self.Q[cur_s[0], cur_s[1], a]
        if next_s is not 'done':
            q_target = r + self.gamma * np.max(self.Q[next_s[0],next_s[1], :])
            self.Q[cur_s[0],cur_s[1], a] += self.lr * (q_target - q_predict) 
        else:
            # next state is terminal q_target = r  
            self.Q[cur_s[0],cur_s[1], a] = r

    # def check_state_exist(self, state):
    #     if list(state) not in self.all_states:
    #         # append new state to q table
    #         self.q_table = self.q_table.append(
    #             pd.Series(
    #                 np.random.uniform(low = -1, high = 1, 
    #                       size = (len(self.actions))),
    #                 index=self.q_table.columns,
    #                 name='+'.join([str(s_i) for s_i in state]),
    #             )
    #         )
    #         self.all_states.append(list(state))