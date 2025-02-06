from .pong_py.pong.game import PongGameContactListener
from .pong_py.pong.paddle import Paddle
from .pong_py.pong.ball import Ball

class TrainPongContactListener(PongGameContactListener):
    """A base collision system listener used for training of agents on Pong."""

    controller_1 = None             #Controller of left paddle.
    controller_2 = None             #Controller of right paddle.

    def BeginContact(self, contact):
        super().BeginContact(contact)

        #Does paddle start contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is the controller_1's paddle?
            if self.controller_1 is not None and self.controller_1.paddle == paddle:
                self.controller_1.is_colliding_ball = True
                self.controller_1.n_touch += 1

            #Is the controller_2's paddle?
            if self.controller_2 is not None and self.controller_2.paddle == paddle:
                self.controller_2.is_colliding_ball = True
                self.controller_2.n_touch += 1

    def EndContact(self, contact):
        super().EndContact(contact)

        #Does paddle end contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is the controller_1's paddle?
            if self.controller_1 is not None and self.controller_1.paddle == paddle:
                self.controller_1.is_colliding_ball = False

            #Is the controller_2's paddle?
            if self.controller_2 is not None and self.controller_2.paddle == paddle:
                self.controller_2.is_colliding_ball = False