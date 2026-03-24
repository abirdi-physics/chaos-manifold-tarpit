#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import StandardScaler

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
        
        return X, y

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
    
def generate_hallucination(model, X_scaler, y_scaler, seed_sequence, steps):
    model.eval()
    hallucinated_path = []
    
    current_window = X_scaler.transform(seed_sequence.reshape(1,-1))
    
    with torch.no_grad():
        for i in range(steps):
            X_tensor = torch.tensor(current_window, dtype = torch.float32)
            
            prediction = model(X_tensor)
            prediction_np = prediction.detach().cpu().numpy()
            
            real_world_points = y_scaler.inverse_transform(prediction_np)
        
            hallucinated_path.append(real_world_points.flatten())
        
            new_window = np.concatenate([current_window[0, 3:], 
                                         prediction_np.flatten()])
            current_window = new_window.reshape(1, -1)
    
    return np.array(hallucinated_path)
    
attractor = Lorenz(1, 1, 1, 2000)
attractor.path()

X_raw, y_raw = attractor.prepare_training_data()

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

X_train = torch.tensor(X_scaled, dtype=torch.float32)
y_train = torch.tensor(y_scaled, dtype=torch.float32)

model = ChaosPredictor()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


num_epochs = 2000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    mse = loss(outputs, y_train)
    mse.backward()
    optimizer.step()
    
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {mse.item()}')

    

    
    
    
    
    
    
    
    