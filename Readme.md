# Mountain car RL task

Env from [gym](https://github.com/openai/gym) and compare different methods on two tasks: [Mountain car v0](https://gym.openai.com/envs/MountainCar-v0) and [Continuous mountain car v0](https://gym.openai.com/envs/MountainCarContinuous-v0)  

## Env

+ Ubuntu  
+ Tensorflow 1.x (test on 1.12.0)  
+ Gym (test on 0.15.4)
+ Numpy (test on 1.17.2)  
+ Matplotlib (test on 3.1.1)

```sh
# or you can directly run: if you use anaconda
$ conda env create -f environment.yml
```

## Mountain Car v0

Result: DQN>Sarsa-lambda>Sarsa~QTable  

### Usage
```sh
$ cd MountainCar
# Run QTable
$ python QTable_MountainCar.py
# Run Sarsa
$ python Sarsa_MountainCar.py
# Run Sarsa-lambda
$ python Sarsalambda_MountainCar.py
# Run DQN
$ python DQN_MountainCar.py
```

## Continuous Mountain Car v0

### A2C
+ 999: state_normalize in 999 iter
+ with_e: state_normalize with self.norm_dist.entropy() 

### A3C
Need better params
+ scratch: no 999 iter constrain with state_normalize, without entropy_beta=0.01 (if with entropy then ... no work)  
+ 999(equal to A3C.png): 999 iter constrain with state_normalize, with entropy_beta=0.01  
+ 999_no_norm: 999 iter constrain without state_normalize, with entropy_beta=0.01 
+ no_e: same to 999, without entropy_beta 