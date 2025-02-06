from pong_pz.env.pong_env import PongParallEnv

from pettingzoo.test import parallel_api_test

def test_env():
    env = PongParallEnv(render_mode="human")
    parallel_api_test(env, num_cycles=100000)