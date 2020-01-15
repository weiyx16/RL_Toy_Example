# Mountain car RL task

Env from [gym](https://github.com/openai/gym) and compare different methods on two tasks: [Mountain car v0](https://gym.openai.com/envs/MountainCar-v0) and [Continuous mountain car v0](https://gym.openai.com/envs/MountainCarContinuous-v0)  

## Env

+ Ubuntu  
+ Tensorflow 1.x (test on 1.12.0)  
+ Gym (test on 0.15.4)
+ Numpy (test on 1.17.2)  
+ Matplotlib (test on 3.1.1)

```sh
# or if you use anaconda, you can directly run: 
$ conda env create -f environment.yml
```

## Usage

**Notice** if you want to render the video, in beginning of each file:
```python
RENDER = True  # Show GUI
```

### Mountain Car v0

Result: DQN>Sarsa-lambda>Sarsa~QTable 

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

### Continuous Mountain Car v0

Result: A2C>A3C

```sh
$ cd MountainCar_Continuous
# Run A2C
$ python A2C_MountainCar.py
# Run A3C
$ python A3C_MountainCar.py
```

## Citation

```bibtex
@misc{RLYixuan2020, 
    author = {Yixuan, Wei},
    title = {Mountain car RL task},
    howpublished = {\url{https://github.com/weiyx16/RL_Toy_Example}},
    year = {2020}
}
```

## Thanks

+ https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow  
+ https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c  
+ https://github.com/stefanbo92/A3C-Continuous  
+ https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python  
+ https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c  
+ https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f  