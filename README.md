# Pong-PettingZoo
## About
This repository contains my personal library which supports 
my project [Pong clone](https://github.com/bottamichele/Pong-Python) for 
[PettingZoo](https://pettingzoo.farama.org/).

## Installation
This library needs to be installed locally to be used. You need to do the following steps to install it:
1. download this repository by running the following command in your console.
   ```
   git clone --recursive https://github.com/bottamichele/Pong-PZ
   ```
2. after you have downloaded the repository, you need to run `pip install <path_repository>`
   (where ***<path_repository>*** is the absolute path of the downloaded repository)
   on terminal to install the library and all its dependecies.

## Usage
Before you run any Pong enviroments, you need to import `pong_pz` and then to create
a new enviroment by using `pong_pz.pong_v0.env()`.

This library contains its wrappers as well and which can be imported from `pong_pz.wrappers`.

The code below is a snippet to get start with this library.
```python
from pong_pz import pong_v0

env = pong_v0.env(render_mode="human")

for episode in range(10):
    episode_ended = False
    observations, infos = env.reset()
    
    while env.agents:
        if not episode_ended:
            actions = {a: env.action_space(a).sample() for a in env.agents}
        else:
            actions = None

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if not episode_ended:
            episode_ended = terminations[env.agents[0]] or truncations[env.agents[1]]

env.close()
```

## About enviroment and wrappers
### Observation
For each agent, observations contains information about positions and velocities of the paddles and the ball. 
An observation is stored into an array of shape 12
```
[agent.position.x, agent.position.y, agent.velocity.x, opponent.velocity.y, opponent.position.x, opponent.position.y, opponent.velocity.x, opponent.velocity.y, ball.position.x, ball.position.y, ball.velocity.x, ball.velocity.y]
```
and states:
- first and second item represent the current position of the paddle which is controlled by the agent.
- third and fourth item represent the current velocity of the paddle which is controlled by the agent.
- fifth and sixth item represent the current position of the paddle which is controlled by the opponent agent.
- seventh and eighth item represent the current velocity of the paddle which is controlled by the opponent agent.
- nineth and tenth item represent the current position of the ball.
- eleventh and twelfth item represent the current velocity of the ball.

### Action
For each agent, the action space supports 3 actions and is:
- **0**: the agent does not move its paddle.
- **1**: the agent moves its paddle up.
- **2**: the paddle moves its paddle down.

### Reward
For each agent, its reward function states:
- **1.0**: when the agent scores a point.
- **-1.0**: when the opponent agent scores a point.
- **0.1**: when the agent touches the ball.
- **0.0**: otherwise.

### `infos` dictionary
The `infos` dictionary contains only debug information of each agent. 
Each agent has its own dictionary which are:
- **'score'**: its current score.
- **'ball_touched'**:' current number of ball touched.

### Wrappers
The wrappers implemented in this library are located in `pong_pz.wrappers` and are:
- **point_reward**: it changes the original reward function of each agent and considers scores as signal rewards.
  The new reward function of each agent states:
  - **1.0**: when the agent scores a point.
  - **-1.0**: when the opponent agent scores a point.
  - **0.0**: otherwise.
- **normalize_observation_pong**: it scales every observation value of each agent between -1.0 and 1.0

## License
This library is licensed under the MIT License. For more information, 
see [LICENSE](https://github.com/bottamichele/Pong-Gym/blob/main/LICENSE).
