# Mountain car RL task

## Task 1

### QTable
+ 200: means stop after 200 iter in one epoch
+ Supervised: reward given when reach higher
+ Weaksupervised: reward given when reach end point

### Sarsa
**Same to QTable**

### SarsaLambda
+ _CleanE: set trace to all zero after each epoch. And set current state&action with 1
+ none: set current s&a += 1

### DQN
Need better params
+ scratch: state_normalize [64,256,128]  #done(r=10),else(r=0)
+ 200: 200 constrain,state_normalize [32,64,32] gamma from 0.9 to 0.99
+ reward_warp_200: 200 constrain,done(r=10),else(r=-1),state_normalize [32,64,32] gamma 0.99

## Task 2

### A2C
+ 999: state_normalize in 999 iter
+ with_e: state_normalize with self.norm_dist.entropy() 

### A3C
Need better params
+ unwarpped: no 999 iter constrain


## Schedule

- [x] DQN on task 1 in leadboard
- [ ] Using another ways to present the **state**, like tile coding in leadboard (and finetune the params)
- [x] Maybe you need to replot graph in QTable
- [x] Task 2 A2C
- [x] Task 2 A3C (40 workers are slower than single one...)
- [ ] Task 2 DDPG
