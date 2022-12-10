import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if larger
        self.model1 = Linear_QNet(11, 256, 3) # n_states, hidden, n_actions
        self.model2 = Linear_QNet(11, 256, 3)  # n_states, hidden, n_actions
        self.trainer = QTrainer(self.model1, self.model2,lr=LR, gamma=self.gamma)


    def get_state(self, game):
        
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Replay memory
        if len(self.memory) > BATCH_SIZE:
            minibatch = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            minibatch = self.memory
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in minibatch:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model1(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores =[]
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    #plt.figure(figsize=(8, 6))
    i = 0
    while True and agent.n_games<111:

        #get old state
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)

        #perform new move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
    
        #train short memory base on the new action and state
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # store the new data into a long term memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done == True:
            # One game is over, train on the memory and plot the result.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model1.save()
                agent.model2.save()
            print('Game', agent.n_games, ', Score:', score, ', Record:', record)
            
            plot_scores.append(score)
            total_score += score
            if agent.n_games > 0:
                mean_score = total_score / agent.n_games
            else:
                mean_score = 0
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)
            # print(plot_scores)
            # print(plot_mean_scores)
            # plt.plot(plot_scores, plot_mean_scores, linestyle="-", linewidth=2, color="k")

            # plot the figure
            # plt.ylabel("Cumulative reward")
            # plt.xlabel("Time step")
            # plt.show()

    return plot_scores, plot_mean_scores



if __name__ == '__main__':
    # plt.plot([0, 0, 0, 0], [0, 3, 65, 6], linestyle="-", linewidth=2, color="k")
    # plt.show(block=True)
    plot_scores, plot_mean_scores = train()
    plt.figure(figsize=(8, 6))
    print("Ravi")
    #plot(plot_scores, plot_mean_scores)
    # plt.plot(plot_scores, plot_mean_scores, linestyle="-", linewidth=2, color="k")
    # plt.show(block=True)
    # print(plot_scores, plot_mean_scores)
    # plt.plot(plot_scores, linestyle="-", linewidth=2, color="k")
    # plt.plot(plot_mean_scores, linestyle="-", linewidth=2, color="y")
    # plt.show(block=True)
    plt.title('Returns')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(plot_scores)
    plt.plot(plot_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
    plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
    plt.show(block=True)
