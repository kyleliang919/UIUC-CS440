from MDP.MDP import MDP
import numpy as np
import random
import math
class Simulator:
    
    def __init__(self, num_games=0, alpha_value=0, gamma_value=0, epsilon_value=0):
        '''
        Setup the Simulator with the provided values.
        :param num_games - number of games to be trained on.
        :param alpha_value - 1/alpha_value is the decay constant.
        :param gamma_value - Discount Factor.
        :param epsilon_value - Probability value for the epsilon-greedy approach.
        '''
        self.num_games = num_games       
        self.epsilon_value = epsilon_value       
        self.alpha_value = alpha_value       
        self.gamma_value = gamma_value
        # Your Code Goes Here!
        self.Q_table= np.zeros((12,12,2,3,12,3))
        self.train_agent()
        totalhit = 0
        #evaluation
        # turn off the exploration
        self.epsilon_value = 0
        for i in range(0,5):
            print("Real game",i)
            totalhit = totalhit + self.play_game()
        print("average hit per game:",totalhit/5)
        #np.set_printoptions(threshold=np.inf)
        #print(self.Q_table)
        
    def f_function(self,b_x,b_y,v_x,v_y,p_y):
        '''
        Choose action based on an epsilon greedy approach
        :return action selected
        '''
        # Your Code Goes Here!
        action_selected = None
        Q_a = np.argmax(self.Q_table[b_x][b_y][v_x][v_y][p_y])
        # epsilon-greedy exploration
        x=random.uniform(0,1)
        if x>=self.epsilon_value:
            action_selected = Q_a
        else:
            action_selected = random.randint(0,2)
        return action_selected

    def train_agent(self):
        '''
        Train the agent over a certain number of games.
        '''
        # Your Code Goes Here!
        alpha=self.alpha_value
        for i in range(0,self.num_games):
            print("trainning game",i)
            mdp=MDP()
            mdp.discretize_state()
            while mdp.discrete_ball_x >= 0:
                print("hit:",mdp.hit)
                b_x = mdp.discrete_ball_x
                b_y = mdp.discrete_ball_y
                v_x = mdp.discrete_velocity_x
                v_y = mdp.discrete_velocity_y
                p_y = mdp.discrete_paddle_y
                s_a = self.f_function(b_x,b_y,v_x,v_y,p_y)
                Q_a = self.Q_table[b_x][b_y][v_x][v_y][p_y][s_a]
                mdp.simulate_one_time_step(s_a)
                mdp.discretize_state()
                r=mdp.reward()
                if r>=0:
                    b_x_new = mdp.discrete_ball_x
                    b_y_new = mdp.discrete_ball_y
                    v_x_new = mdp.discrete_velocity_x
                    v_y_new = mdp.discrete_velocity_y
                    p_y_new = mdp.discrete_paddle_y
                    idx = self.Q_table[b_x_new][b_y_new][v_x_new][v_y_new][p_y_new].argmax()
                    Q_next_max =  self.Q_table[b_x_new][b_y_new][v_x_new][v_y_new][p_y_new][idx]
                else:
                    #Q_next_max = self.max_Q(self.Q_table[6][6][1][2][6])
                    Q_next_max = 0
                self.Q_table[b_x][b_y][v_x][v_y][p_y][s_a] = Q_a+alpha*(r+self.gamma_value*Q_next_max-Q_a)
            #alpha = self.alpha_value*(1-i*alpha/self.num_games)
            #alpha = self.alpha_value/(i+1)
        pass
    def play_game(self):
        '''
        Simulate an actual game till the agent loses.
        '''
        # Your Code Goes Here!
        mdp = MDP()
        mdp.discretize_state()
        while mdp.discrete_ball_x >= 0:
            b_x = mdp.discrete_ball_x
            b_y = mdp.discrete_ball_y
            v_x = mdp.discrete_velocity_x
            v_y = mdp.discrete_velocity_y
            p_y = mdp.discrete_paddle_y
            mdp.simulate_one_time_step(self.f_function(b_x,b_y,v_x,v_y,p_y))
            mdp.discretize_state()
        print ("final hits:",mdp.hit)
        return mdp.hit