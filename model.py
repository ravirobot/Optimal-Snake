import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear15 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        #x = self.linear1(x)
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear15(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    
    def __init__(self, model1,model2, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model1 = model1
        self.model2 = model2
        self.optimizer = optim.Adam(model1.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # (1, n)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        # 1: predicted Q values with current state
        pred = self.model1(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            # 2a: r + y * next precited Q value
            Q_new1 = reward[idx]
            Q_new2 = reward[idx]
            if random.random()>0:
                update_a=0
            else:
                update_a=1
            if not done[idx]:
                if update_a == 0:
                    Q_new1 = reward[idx] + self.gamma * torch.max(self.model2(next_state[idx]))
                    target[idx][torch.argmax(action[idx]).item()] = Q_new1
                else:
                    Q_new2 = reward[idx] + self.gamma * torch.max(self.model1(next_state[idx]))
                    target[idx][torch.argmax(action[idx]).item()] = Q_new2
                # r + discounted_reward           
            
            # 2b: preds[argmax(action)] = Q_new
            #target[idx][torch.argmax(action[idx]).item()] = Q_new1
        
        # 1: preds = predicted Q values with current state
        # 2: r + y * next precited Q value
        #    preds[argmax(action)] = Q_new
         
        self.optimizer.zero_grad()
            
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()