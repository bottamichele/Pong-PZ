from pong_pz.env.pong_env import PongParallEnv
from pong_pz.wrappers import normalize_observation_pong

def test_normalize_obs_pong():
    env = PongParallEnv(render_mode="human")
    env = normalize_observation_pong(env)

    while True:
        r1 = 0
        r2 = 0
        env.reset()
        done = False

        while env.agents:
            if done:
                break

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            observations, rewards, terminations, truncations, infos = env.step(actions)
            r1 += rewards[env.agents[0]]
            r2 += rewards[env.agents[1]]

            done = terminations[env.agents[0]] or terminations[env.agents[1]]

            print(f"- obss = {observations}")
    env.close()