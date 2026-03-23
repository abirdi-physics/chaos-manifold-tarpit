#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:47:10 2026

@author: alexandre
"""
import numpy as np

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
            
            if (i+ 1) % 100 == 0:
                self.manifold.append((self.x, self.y, self.z))
                
        return self.trajectory
            
    def distance(self, point1, point2):
        
        return np.sqrt((point1[0] - point2[0])**2
                       + (point1[1] - point2[1]) ** 2
                       + (point1[2] - point2[2]) ** 2)


