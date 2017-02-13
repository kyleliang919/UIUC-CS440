import random
import math
class MDP:
    
    def __init__(self, 
                 ball_x=None,
                 ball_y=None,
                 velocity_x=None,
                 velocity_y=None,
                 paddle_y=None):
        '''
        Setup MDP with the initial values provided.
        '''
        self.create_state(
            ball_x=ball_x,
            ball_y=ball_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            paddle_y=paddle_y
        )
        
        # the agent can choose between 3 actions - stay, up or down respectively.
        self.actions = [0, 0.04, -0.04]
        # record how many hits are done so far
        self.hit = 0
        # flag for whether the ball is going to hit in next state
        self.hitting = False
        self.discrete_ball_x = 6
        self.discrete_ball_y = 6
        self.discrete_velocity_x = 1
        self.discrete_velocity_y = 2
        self.discrete_paddle_y = 6
    
    def create_state(self,
              ball_x=None,
              ball_y=None,
              velocity_x=None,
              velocity_y=None,
              paddle_y=None):
        '''
        Helper function for the initializer. Initialize member variables with provided or default values.
        '''
        self.paddle_height = 0.2
        self.ball_x = ball_x if ball_x != None else 0.5
        self.ball_y = ball_y if ball_y != None else 0.5
        self.velocity_x = velocity_x if velocity_x != None else 0.03
        self.velocity_y = velocity_y if velocity_y != None else 0.01
        self.paddle_y = 0.5-self.paddle_height/2
       
        
    
    def simulate_one_time_step(self, action_selected):
        '''
        :param action_selected - Current action to execute.
        Perform the action on the current continuous state.
        '''
        # Your Code Goes Here!
        #Update the position of the ball and the paddle
        self.ball_x = self.ball_x + self.velocity_x
        self.ball_y = self.ball_y + self.velocity_y
        self.paddle_y = self.paddle_y + self.actions[action_selected] 
        if self.paddle_y < 0:
            self.paddle_y = 0
        if self.paddle_y > 0.8:
            self.paddle_y = 0.8
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        if self.ball_y > 1:
            self.ball_y = 2 -self.ball_y
            self.velocity_y = -self.velocity_y
        if self.ball_x < 0:
            self.ball_x = -self.ball_x
            self.velocity_x = -self.velocity_x
        if self.ball_x > 1:
            if self.ball_y > self.paddle_y and self.ball_y < self.paddle_y + self.paddle_height:
                self.ball_x = 2 - self.ball_x
                U = random.uniform(-0.015,0.015) 
                V = random.uniform(-0.03,0.03)
                if abs(-self.velocity_x + U) < 1 and abs(-self.velocity_x + U) > 0.03:
                    self.velocity_x= -self.velocity_x + U
                else:
                    self.velocity_x= -self.velocity_x
                if abs(self.velocity_y + V) < 1:
                    self.velocity_y= self.velocity_y + V
                self.hit = self.hit + 1
                self.hitting = True
        pass
    
    def discretize_state(self):
        '''
        Convert the current continuous state to a discrete state.
        '''
        # Your Code Goes Here!
        if(self.ball_x > 1):
            self.discrete_ball_x = -1
        else:
            self.discrete_ball_x = math.floor(self.ball_x*12)
            if self.ball_x == 1:
                self.discrete_ball_x = 11
            self.discrete_ball_y = math.floor(self.ball_y*12)
            if self.ball_y == 1:
                self.discrete_ball_y = 11
            self.discrete_velocity_x = 1 if self.velocity_x > 0 else 0
            if abs(self.velocity_y) < 0.015:
                self.discrete_velocity_y = 1
            else:
                self.discrete_velocity_y = 2 if self.velocity_y > 0 else 0
            if 1-self.paddle_height == self.paddle_y:
                self.discrete_paddle_y = 11
            else:
                self.discrete_paddle_y = math.floor(12*self.paddle_y/(1-self.paddle_height)) 
        pass
    
    def reward(self):
        if self.discrete_ball_x < 0:
            return -1
        elif self.hitting:
            self.hitting = False
            return 1
        else:
            return 0