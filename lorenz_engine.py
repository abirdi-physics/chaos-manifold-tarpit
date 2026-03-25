#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Lorenz(): #Physics of Lorenz attractor
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
    
    def path(self): #Creates trajectory for initial condition
        
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
            
    def distance(self, point1, point2): #Finds distance between points
        
        return np.sqrt((point1[0] - point2[0])**2
                       + (point1[1] - point2[1]) ** 2
                       + (point1[2] - point2[2]) ** 2) 
    
    def verify_match(self, manifold): #verifies if point belongs to trajectory
        
        for p1, p2 in zip(self.manifold, manifold):
            
            if self.distance(p1, p2) > 0.02:
                return False
            
        return True
    
    def prepare_training_data(self, window_size = 3): #Prepares data from self.trajectory
        X = []  
        y = []

        for i in range(len(self.trajectory) - window_size):
#Creates window and flattens arrays for pytorch
            X.append(self.trajectory[i:i + window_size])
            y.append(self.trajectory[i + window_size])
        X = np.array(X)
        X = X.reshape(X.shape[0], -1)
        y = np.array(y)
        
        return X, y

class ChaosPredictor(nn.Module): #Neural Network
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
#neural network tries to predict a trajectory
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
#gives R^2 of regression
def regression_score(model, X_test, y_test, y_scaler):
    model.eval()      
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype = torch.float)
        predictions = model(X_test)
        predictions_np = predictions.detach().cpu().numpy()
        descaled_predictions = y_scaler.inverse_transform(predictions_np)
        descaled_y = y_scaler.inverse_transform(y_test)
        
        return r2_score(descaled_y, descaled_predictions)
            
    
attractor = Lorenz(1, 1, 1, 2000)
attractor.path()

X_raw, y_raw = attractor.prepare_training_data()

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled
                                                    , test_size = 0.2
                                                    , shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

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

print(regression_score(model, X_test, y_test, y_scaler))
    

    
    
    
    
    
    
    
    