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

## Schedule

- [ ] DQN on task 1 in leadboard
- [ ] Using another ways to present the **state**, like tile coding in leadboard (and finetune the params)
- [ ] Maybe you need to replot graph in QTable
- [x] Task 2 A2C
- [ ] Task 2 DDPG
