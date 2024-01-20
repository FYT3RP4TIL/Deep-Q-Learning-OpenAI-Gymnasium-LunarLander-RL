# Deep-Q-Learning-OpenAI-Gymnasium-LunarLander
![1](https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/0b54cd19-5197-49f2-93da-80f5f205a088)
## Environment : Credits - Oleg Klimov

| Action Space | ``` Discrete(4) ``` |
| :---   | :--- | 
| Observation |  ``` Box([-1.5 -1.5 -5. -5. -3.1415927 -5. -0. -0. ], [1.5 1.5 5. 5. 3.1415927 5. 1. 1. ], (8,), float32) ``` | 
| Import | ``` gymnasium.make("LunarLander-v2") ``` |

## Description
This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

## Action Space
There are four discrete actions available:

* 0: do nothing
* 1: fire left orientation engine
* 2: fire main engine
* 3: fire right orientation engine

## Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in ```x``` & ```y```, its linear velocities in ```x``` & ```y```, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Rewards
After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:

* is increased/decreased the closer/further the lander is to the landing pad.
* is increased/decreased the slower/faster the lander is moving.
* is decreased the more the lander is tilted (angle not horizontal).
* is increased by 10 points for each leg that is in contact with the ground.
* is decreased by 0.03 points each frame a side engine is firing.
* is decreased by 0.3 points each frame the main engine is firing.
* The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
* An episode is considered a solution if it scores at least 200 points.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points.

## Starting State
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

## Episode Termination
The episode finishes if:

* 1.the lander crashes (the lander body gets in contact with the moon);
* 2.the lander gets outside of the viewport (x coordinate is greater than 1);
* 3.the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:

```When Box2D determines that a body (or group of bodies) has come to rest, the body enters a sleep state which has very little CPU overhead. If a body is awake and collides with a sleeping body, then the sleeping body wakes up. Bodies will also wake up if a joint or contact attached to them is destroyed.```

## Arguments
To use to the continuous environment, you need to specify the continuous=True argument like below:

import gymnasium as gym
env = gym.make(
    "LunarLander-v2",
    continuous: bool = False,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
)
If ```continuous=True``` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the action space will be ```Box(-1, +1, (2,), dtype=np.float32)```. The first coordinate of an action determines the throttle of the main engine, while the second coordinate specifies the throttle of the lateral boosters. Given an action ```np.array([main, lateral])```, the main engine will be turned off completely if main < 0 and the throttle scales affinely from 50% to 100% for ```0 <= main <= 1``` (in particular, the main engine doesn’t work with less than 50% power). Similarly, if ```-0.5 < lateral < 0.5```, the lateral boosters will not fire at all. If ```lateral < -0.5```, the left booster will fire, and if ```lateral > 0.5```, the right booster will fire. Again, the throttle scales affinely from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

```gravity``` dictates the gravitational constant, this is bounded to be within 0 and -12.

If ```enable_wind=True``` is passed, there will be wind effects applied to the lander. The wind is generated using the function ```tanh(sin(2 k (t+C)) + sin(pi k (t+C)))```. k is set to 0.01. ```C``` is sampled randomly between -9999 and 9999.

```wind_power``` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for ```wind_power``` is between 0.0 and 20.0. ```turbulence_power``` dictates the maximum magnitude of rotational wind applied to the craft. The recommended value for ```turbulence_power``` is between 0.0 and 2.0.

## Version History
* v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters (used) 

* v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.

* v0: Initial version

## Notes
There are several unexpected bugs with the implementation of the environment.

* 1.The position of the side thursters on the body of the lander changes, depending on the orientation of the lander. This in turn results in an orientation depentant torque being applied to the lander.

* 2.The units of the state are not consistent. I.e.
   * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs 
     to be multiplied by a factor of 2.5.

For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal: ‘x’: 10 ‘y’: 6.666 ‘vx’: 5 ‘vy’: 7.5 ‘angle’: 1 ‘angular velocity’: 2.5

After the correction has been made, the units of the state are as follows: ‘x’: (units) ‘y’: (units) ‘vx’: (units/second) ‘vy’: (units/second) ‘angle’: (radians) ‘angular velocity’: (radians/second)

## Deep - Q - Learning :--

### The DQN Agent
The [DQN (Deep Q-Network) algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) was developed by DeepMind in 2015. It was able to solve a wide range of Atari games (some to superhuman level) by combining reinforcement learning and deep neural networks at scale. The algorithm was developed by enhancing a classic RL algorithm called Q-Learning with deep neural networks and a technique called experience replay.

### Loss Function
![Screenshot 2024-01-19 165706](https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/cba64cb3-d093-41ce-aa24-caba81409723)

### Experience Replay
The second major addition to make DQNs work is Experience Replay. The basic idea is that by storing an agent’s experiences, and then randomly drawing batches of them to train the network, we can more robustly learn to perform well in the task. By keeping the experiences we draw random, we prevent the network from only learning about what it is immediately doing in the environment, and allow it to learn from a more varied array of past experiences.

![Role-of-experience-replay-in-deep-reinforcement-learning](https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/a83ceebb-7ab0-47c5-b191-d33f79e49f25)

### Separate Target Network
The third major addition to the DQN that makes it unique is the utilization of a second network during the training procedure. This second network is used to generate the target-Q values that will be used to compute the loss for every action during training. 

### Action Selection Policy
From this problem has emerged an entire subfield within reinforcement learning that has attempted to develop techniques for meaningfully balancing the exploration and exploitation tradeoff. Ideally, such an approach should encourage exploring an environment until the point that it has learned enough about it to make informed decisions about optimal actions. There are a number of frequently used approaches to encouraging exploration. Here we will use [Epsilon-Greedy](https://www.baeldung.com/cs/epsilon-greedy-q-learning).

![egreedy](https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/2692d7ea-414a-4ca2-a9ad-a4facd4008aa)

Each value corresponds to the Q-value for a given action at a random state in an environment. The height of the light blue bars correspond to the probability of choosing a given action. The dark blue bar corresponds to a chosen action.

**Explanation** : A simple combination of the greedy and random approaches yields one of the most used exploration strategies: ϵ-greedy. In this approach the agent chooses what it believes to be the optimal action most of the time, but occasionally acts randomly.

## Results : AI landing Itself

https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/e1d99825-412c-4596-bfb2-ca2b686fc317

## References
* https://gymnasium.farama.org/index.html
* https://www.tensorflow.org/agents/tutorials/0_intro_rl
* https://medium.com/intro-to-artificial-intelligence/deep-q-network-dqn-applying-neural-network-as-a-functional-approximation-in-q-learning-6ffe3b0a9062
* https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
* https://arxiv.org/pdf/1511.05952.pdf 
* http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf

