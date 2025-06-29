import numpy as np
import functools
import pygame

from pettingzoo import ParallelEnv

from gymnasium.spaces import Discrete, Box

from pygame import Vector2, Rect

from .pong_py.pong.ball import Ball
from .pong_py.pong.paddle import Paddle
from .pong_py.pong.game import Game
from .pong_py.pong.controller.controller import PaddlePosition, MovingType

from .agent_controller import AgentController
from .train_pong_cl import TrainPongContactListener

class PongParallEnv(ParallelEnv):
    """A parallel enviroment that support Pong clone game for Petting Zoo."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        """Create a new Pong enviroment.
        
        Parameter
        --------------------
        render_mode: str, optional
            render mode to use"""
        
        assert render_mode is None or render_mode in self.metadata["render_modes"], "render_mode supports None, \"rgb_array\" and \"human\"."

        #Public attributes.
        self.render_mode = render_mode
        self.agents = []
        self.possible_agents = ["paddle_1", "paddle_2"]
        
        #Window parameters.
        self._window_size = (700, 550)
        self._window = None
        self._clock = None
        self._font = None
        self._font_color = (255, 255, 255)

        #Pong's game parameters.
        self._current_game = None
        self._agent_1_controller = None
        self._agent_2_controller = None
        self._last_agent_1_score = 0
        self._last_agent_2_score = 0

        #Observation space.
        temp_game = Game()
        self._observation_space = Box(low=np.array([temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   0,
                                                   -Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   0,
                                                   -Paddle.SPEED,
                                                   temp_game.field.center_position.x - temp_game.field.width/2,
                                                   temp_game.field.center_position.y - temp_game.field.height/2,
                                                   -Ball.SPEED,
                                                   -Ball.SPEED], 
                                                   dtype=np.float32), 
                                    high=np.array([temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   0,
                                                   Paddle.SPEED,
                                                   temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   0,
                                                   Paddle.SPEED,
                                                   temp_game.field.center_position.x + temp_game.field.width/2,
                                                   temp_game.field.center_position.y + temp_game.field.height/2,
                                                   Ball.SPEED,
                                                   Ball.SPEED], 
                                                   dtype=np.float32), 
                                    shape=(12,), 
                                    dtype=np.float32)        

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
        
    def _get_obs(self):
        """Return the current observations of each player."""
        
        assert self._current_game is not None, "No game is started."
        
        paddle_1_pos = self._current_game.paddle_1.position
        paddle_1_vel = self._current_game.paddle_1.velocity
        paddle_2_pos = self._current_game.paddle_2.position
        paddle_2_vel = self._current_game.paddle_2.velocity
        ball_pos = self._current_game.ball.position
        ball_vel = self._current_game.ball.velocity
        
        return {self.agents[0]: np.array([paddle_1_pos.x, paddle_1_pos.y, 
                                          paddle_1_vel.x, paddle_1_vel.y,
                                          paddle_2_pos.x, paddle_2_pos.y,
                                          paddle_2_vel.x, paddle_2_vel.y,
                                          ball_pos.x, ball_pos.y,
                                          ball_vel.x, ball_vel.y]),
                self.agents[1]: np.array([paddle_2_pos.x, paddle_2_pos.y,
                                          paddle_2_vel.x, paddle_2_vel.y,
                                          paddle_1_pos.x, paddle_1_pos.y, 
                                          paddle_1_vel.x, paddle_1_vel.y,
                                          ball_pos.x, ball_pos.y,
                                          ball_vel.x, ball_vel.y])}
    
    def _get_info(self):
        """Return a infos dict."""

        assert self._current_game is not None, "No game is started."
        assert self._agent_1_controller is not None, "controller_1 does not have any controllers."
        assert self._agent_2_controller is not None, "controller_2 does not have any controllers."

        return {self.agents[0]: {"score": self._current_game.score_paddle_1, "ball_touched": self._agent_1_controller.n_touch}, 
                self.agents[1]: {"score": self._current_game.score_paddle_2, "ball_touched": self._agent_2_controller.n_touch}}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        #Create and start a new game.
        cl = TrainPongContactListener()
        self._current_game = Game(contact_listener=cl)
        self._current_game.start()

        #Controllers are set.
        self._agent_1_controller = AgentController(self._current_game.paddle_1, PaddlePosition.LEFT)
        self._agent_2_controller = AgentController(self._current_game.paddle_2, PaddlePosition.RIGHT)
        cl.controller_1 = self._agent_1_controller
        cl.controller_2 = self._agent_2_controller

        #Initialize last player scores.
        self._last_agent_1_score = self._current_game.score_paddle_1
        self._last_agent_2_score = self._current_game.score_paddle_2

        #Return the initial observation.
        return self._get_obs(), self._get_info()

    def step(self, actions):
        if actions is None:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        #Store actions to perform into agent controllers.
        self._agent_1_controller.set_next_move(MovingType(actions[self.agents[0]]))
        self._agent_2_controller.set_next_move(MovingType(actions[self.agents[1]]))

        #Perform an update step.
        self._agent_1_controller.update(1 / self.metadata["render_fps"])
        self._agent_2_controller.update(1 / self.metadata["render_fps"])
        self._current_game.update(1 / self.metadata["render_fps"])

        #Rewards.
        current_agent_1_score = self._current_game.score_paddle_1
        current_agent_2_score = self._current_game.score_paddle_2

        if current_agent_1_score > self._last_agent_1_score:        #Has left agent got a point?
            agent_1_reward = 1.0
            agent_2_reward = -1.0
        elif current_agent_2_score > self._last_agent_2_score:      #Has right agent got a point?
            agent_1_reward = -1.0
            agent_2_reward = 1.0
        else:
            agent_1_reward = 0.1 if self._agent_1_controller.is_colliding_ball else 0.0
            agent_2_reward = 0.1 if self._agent_2_controller.is_colliding_ball else 0.0

        #Update last player scores.
        self._last_agent_1_score = current_agent_1_score
        self._last_agent_2_score = current_agent_2_score

        if self.render_mode == "human":
            self.render()

        return  self._get_obs(), \
                {self.agents[0]: agent_1_reward, self.agents[1]: agent_2_reward }, \
                {self.agents[0]: self._current_game.is_ended(), self.agents[1]: self._current_game.is_ended()}, \
                {self.agents[0]: False, self.agents[1]: False}, \
                self._get_info()
    
    def render(self):
        if self.render_mode is None:
            return
        
        #Initialize font.
        if self._font is None:
            pygame.init()
            self._font = pygame.font.Font(None, 50)
        
        #Initialize window and clock if render_mode is "human".
        if self.render_mode == "human":
            if self._window is None:
                pygame.display.init()
                pygame.display.set_caption("Pong")

                self._window = pygame.display.set_mode(self._window_size)
            if self._clock is None:
                self._clock = pygame.time.Clock()

        #Create new canvas and fill with black color.
        canvas = pygame.Surface(self._window_size)
        canvas.fill("black")

        #Draw text.
        self._draw_score(canvas, self._current_game.score_paddle_1, (self._window_size[0] // 4, 25))
        self._draw_score(canvas, self._current_game.score_paddle_2, (3 * self._window_size[0] // 4, 25))

        #Draw the borders of field.
        self._draw_border_field(canvas)

        #Draw paddles and ball on canvas.
        self._draw_rect(canvas, self._current_game.paddle_1.position, self._current_game.paddle_1.width, self._current_game.paddle_1.height)
        self._draw_rect(canvas, self._current_game.paddle_2.position, self._current_game.paddle_2.width, self._current_game.paddle_2.height)
        self._draw_rect(canvas, self._current_game.ball.position, self._current_game.ball.radius, self._current_game.ball.radius)

        if self.render_mode == "human":
            self._window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            pygame.event.pump()
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _translate_position(self, position):
        """Translate from position of an object to canvas coordinate."""
        
        new_x = position.x + self._window_size[0]/2
        new_y = -(position.y - self._window_size[1]/2)

        return Vector2(new_x, new_y)

    def _draw_score(self, canvas, score_paddle, position):
        """Draw score of a paddle on canvas."""
    
        #Text
        score_paddle_text = self._font.render("{}".format(score_paddle), True, self._font_color)
        
        #Text position on screen
        score_paddle_rect = score_paddle_text.get_rect()
        score_paddle_rect.center = position

        #Draw text
        canvas.blit(score_paddle_text, score_paddle_rect)

    def _draw_border_field(self, canvas, height=20):
        """Draw the borders of field on canvas."""
        
        #Top border of field.
        top_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y + self._current_game.field.height/2))
        pygame.draw.rect(canvas, "white", Rect(top_border_pos.x, top_border_pos.y - height, self._window_size[0], height))

        #Bottom border of field.
        bottom_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y - self._current_game.field.height/2))
        pygame.draw.rect(canvas, "white", Rect(bottom_border_pos.x, bottom_border_pos.y, self._window_size[0], height))

    def _draw_rect(self, canvas, position, width, height):
        """Draw a rectangle on canvas."""
    
        left_vertix_pos = self._translate_position(position + Vector2(-width/2, height/2))
        pygame.draw.rect(canvas, "white", Rect(left_vertix_pos.x, left_vertix_pos.y, width, height))

    def close(self):
        if self._window is not None:
            pygame.display.quit()

        if self._font is not None:
            pygame.quit()