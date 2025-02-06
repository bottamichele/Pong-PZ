from pong_pz.env.pong_env import PongParallEnv
from pong_pz.wrappers import point_reward

def test_point_reward():
    env = PongParallEnv(render_mode="human")
    env = point_reward(env)
    
    while True:
        infos = {}
        r1 = 0
        r2 = 0
        env.reset()
        done = False

        while env.agents:
            if done:
                break

            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            observations, rewards, terminations, truncations, infos = env.step(actions)
            r1 += rewards[env.agents[0]]
            r2 += rewards[env.agents[1]]

            done = terminations[env.agents[0]] or terminations[env.agents[1]]

        print(f"- infos = {infos};  r1 = {r1}; r2 = {r2}")
    env.close()