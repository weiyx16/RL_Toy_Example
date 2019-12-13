"""
Environment for MountainCar-V0 from gym
Src: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

Requirement:
Tensorflow: 2.0
gym: 0.15.4

https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""

import gym
import numpy as np
from Q_learning import QLearningTable
import matplotlib.pyplot as plt

def state_discrete(state):
    # discrete position with 0.1 and velocity with 0.01 and rescale to positive
    # -> state_space (19, 15)
    # ref: https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f
    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    return state_adj

def my_reward(next_state, done, reward, iepoch_max_atti):
    """
        ref: https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
        src reward is -1 in all set.
    """
    if next_state[0] > iepoch_max_atti:
        reward += 11
    # if done:
    #     reward += 2
    return reward

## Environment
RENDER = False  # Show GUI
env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible
env = env.unwrapped  # remove the 200 time step limit the cart pole example defaults to

print('-'*30+'\r\nAction Space: {}\r\n'.format(env.action_space)) # spaces.Discrete(3)
print('-'*30+'\r\nObservation Space: {}\r\n'.format(env.observation_space)) 
# self.state = (position, velocity) range:[-1.2,-0.07]~[0.6,0.07]

## Agent
Agent = QLearningTable(
    actions=np.arange(env.action_space.n), # 3
    states=state_discrete(env.observation_space.high) + 1,
    learning_rate=0.2,
    gamma=0.9,
    epsilon=1,
    begin_epsilon=0.2
)

## Training
epoch = 5000
log_frequency = 100
episode_rewards = []
episode_attitude = []
for iepoch in range(epoch):

    cur_state = state_discrete(env.reset())

    action_count, sum_reward, iepoch_max_atti = 0, 0.0, env.observation_space.low[0]

    done = False
    while not done:
        if RENDER: env.render()

        action = Agent.choose_action(cur_state, iepoch/epoch)

        next_state, reward, done, _ = env.step(action)
        iepoch_max_atti = max(iepoch_max_atti, next_state[0])
        action_count += 1
        sum_reward += reward

        if done and next_state[0] >= 0.5:  # in case of done when iter reach 200
            Agent.learn(cur_state, action, my_reward(next_state, done, reward, iepoch_max_atti), 'done')  
            next_state = state_discrete(next_state)
        else:
            Agent.learn(cur_state, action, my_reward(next_state, done, reward, iepoch_max_atti), state_discrete(next_state))
            next_state = state_discrete(next_state)
            cur_state = next_state
    
    episode_rewards.append(sum_reward)
    episode_attitude.append(iepoch_max_atti)
    # if (iepoch+1) % log_frequency == 0:
    print("episode:", iepoch, " action tried:", int(action_count), " Heighest:",iepoch_max_atti, " average reward to now:",np.mean(episode_rewards))

env.close()
plt.plot(episode_rewards)
plt.plot(episode_attitude)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('./img/QTable_Reward_Supervised.png')
plt.close()