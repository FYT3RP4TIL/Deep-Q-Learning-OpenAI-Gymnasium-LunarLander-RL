# Deep-Q-Learning-OpenAI-Gymnasium-LunarLander
![1](https://github.com/FYT3RP4TIL/Deep-Q-Learning-OpenAI-Gymnasium-LunarLander-RL/assets/113416452/0b54cd19-5197-49f2-93da-80f5f205a088)
## Environment :
| Action Space | ``` Discrete(4) ``` |
| :---:   | :--- | 
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
