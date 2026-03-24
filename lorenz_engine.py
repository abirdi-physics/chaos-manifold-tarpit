#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:47:10 2026

@author: alexandre
"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim

class Lorenz():
    def __init__(self, x, y, z, steps):
        self.sigma = 10
        self.rho = 28
        self.beta = 8.0/3.0
        self.steps = steps
        
        self.dt = 0.01
        self.x = x
        self.y = y
        self.z = z
        
        self.trajectory = []
        self.manifold = []
    
    def path(self):
        
        for i in range(self.steps):
            
            dx = self.sigma * (self.y - self.x)
            dy = self.x *  (self.rho - self.z) - self.y
            dz = (self.x * self.y) - (self.beta * self.z)
            
            self.x = self.x + (dx * self.dt)
            self.y = self.y + (dy * self.dt)
            self.z = self.z + (dz * self.dt)
            
            self.trajectory.append((self.x, self.y, self.z))
            
            if (i + 1) % 100 == 0:
                self.manifold.append((self.x, self.y, self.z))
                
        return self.trajectory
            
    def distance(self, point1, point2):
        
        return np.sqrt((point1[0] - point2[0])**2
                       + (point1[1] - point2[1]) ** 2
                       + (point1[2] - point2[2]) ** 2)
    
    def verify_match(self, manifold):
        
        for p1, p2 in zip(self.manifold, manifold):
            
            if self.distance(p1, p2) > 0.02:
                return False
            
        return True
    
    def prepare_training_data(self, window_size = 3):
        X = []
        y = []

        for i in range(len(self.trajectory) - window_size):

            X.append(self.trajectory[i:i + window_size])
            y.append(self.trajectory[i + window_size])
        X = np.array(X)
        X = X.reshape(X.shape[0], -1)
        y = np.array(y)
        
        return torch.tensor(X, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)

class ChaosPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,3))
        
    def forward(self, x):
        return self.model(x)
    
attractor = Lorenz(1, 1, 1, 200)
attractor.path()

X_train, y_train = attractor.prepare_training_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = ChaosPredictor()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

X_train = X_train.to(device)
y_train = y_train.to(device)

num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    mse = loss(outputs, y_train)
    mse.backward()
    optimizer.step()
    
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {mse.item()}')
        
        
        
        
        
        
        
        
        
        
        
        