from pettingzoo.utils import BaseParallelWrapper

def point_reward(env):
    """Make only scores as signal rewards."""
    
    return PointReward(env)

class PointReward(BaseParallelWrapper):
    """A reward wrapper which considers only scores as signal rewards."""

    def step(self, actions):
        obss, rewards, terminated, truncated, infos = self.env.step(actions)

        if actions is not None:
            for a in self.agents:
                r = rewards[a]
            
                if r != 1.0 and r != -1.0:
                    r = 0.0

                rewards[a] = r

        return obss, rewards, terminated, truncated, infos