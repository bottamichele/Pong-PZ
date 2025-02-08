from .env.pong_env import PongParallEnv

def env(render_mode=None):
    return PongParallEnv(render_mode=render_mode)