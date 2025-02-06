import numpy as np

from gymnasium.spaces import Box
from pettingzoo.utils import BaseParallelWrapper

from ..env.pong_py.pong.paddle import Paddle

def normalize_observation_pong(env):
    """Normalize observation of the Pong game."""
    
    return NormalizeObservationPong(env)

class NormalizeObservationPong(BaseParallelWrapper):
    """A wrapper which normalizes observations of the Pong game."""

    def __init__(self, env):
        super().__init__(env)

        #Field parameters.
        self._field_center = (self.unwrapped.observation_space(self.possible_agents[0]).low[[0, 1]] + \
                              self.unwrapped.observation_space(self.possible_agents[0]).high[[0, 1]]) / 2
        self._field_size = np.abs(self.unwrapped.observation_space(self.possible_agents[0]).high[[0, 1]] - \
                                  self.unwrapped.observation_space(self.possible_agents[0]).low[[0, 1]])

    def observation_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def _normalize_obs(self, obs):
        obs_norm = np.zeros(12, dtype=np.float32)

        #Normalize the ball and paddle positions.
        for idx_obj_pos in [[0, 1], [4, 5], [8, 9]]:
            obs_norm[idx_obj_pos] = (obs[idx_obj_pos] - self._field_center) / (self._field_size/2)

        #Normalize the paddle velocity y.
        for idx_paddle_vel_y in [3, 7]:
            obs_norm[idx_paddle_vel_y] = obs[idx_paddle_vel_y] / Paddle.SPEED

        #Normalize the ball velocity.
        obs_norm[[10, 11]] = obs[[10, 11]] / np.linalg.norm(obs[[10, 11]])

        return obs_norm

    def reset(self, seed=None, options=None):
        obss, infos = self.env.reset(seed=seed, options=options)

        for a in self.agents:
            obs = obss[a]
            obss[a] = self._normalize_obs(obs)

        return obss, infos
    
    def step(self, actions):
        next_obss, rewards, terminated, truncated, infos = super().step(actions)

        if actions is not None:
            for a in self.agents:
                next_obs = next_obss[a]
                next_obss[a] = self._normalize_obs(next_obs)

        return next_obss, rewards, terminated, truncated, infos