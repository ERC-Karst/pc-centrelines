# -*- coding: utf-8 -*-
"""
ASEN 5519 Algorithmic Motion Plannig
Homework Set 2 
Question 2

Program to calculate the configuration space obstacle for a robot which can translate and rotate.
Robot has the same shape as that of a traingular obstacle with vertices (0,0), (1,2) and (0,2)

@author: Shrivatsan K Chari
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from matplotlib.colors import ListedColormap
from shapely.geometry import box
from shapely.ops import nearest_points
from shapely import affinity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def slope_line(p1,p2): #takes in two points and finds the slope of line joining them
    m = p2-p1
    if np.arctan2(m[1],m[0]) >= 0:
        return np.arctan2(m[1],m[0])
    else:
        return 2*np.pi + np.arctan2(m[1],m[0])


thetha= np.arange(0,2*np.pi,(np.pi/180))    #Discritizing the thrid angle parameter
thetha = thetha*(180/np.pi)
W = Polygon([(0,0),(1,2),(0,2)])    #Obstacle
W_vertices = []
for point in list(W.exterior.coords):
    (x,y) = point
    W_vertices.append(np.array([x,y]))
   
angle_W_vertices = []    #Calculating angle of transition from one vertice to another for the obstacle
for i in range(len(W_vertices)):
    if i == len(W_vertices)-1:
        angle_W_vertices.append(slope_line(W_vertices[i], W_vertices[0]))
    else:
        angle_W_vertices.append(slope_line(W_vertices[i], W_vertices[i+1]))
                     
     

minkowski_sum_collection = []

for angle in thetha:
        
        V = affinity.rotate(W, angle) 
        V_prime = affinity.rotate(V, 180) # Finding negative of robot,so that we can find minkowski sum rather than difference
        
        
        V_prime_vertices = [] # Collecting vertices in a list
        for point in list(V_prime.exterior.coords):
            (x,y) = point
            V_prime_vertices.append(np.array([x,y]))
        
        
        minimum = V_prime_vertices[0][1]
        
        for i in range(len(V_prime_vertices)):  # Sorting vertices, starting from lowest ordinate value
            if V_prime_vertices[i][1] < minimum:
                minimum_index = i
        V_prime_vertices = V_prime_vertices[minimum_index:] + V_prime_vertices[:minimum_index]
            
        angle_V_prime_vertices = []    #Calculating angle of transition from one vertice to another for the negative of robot
        for i in range(len(V_prime_vertices)):
            if i == len(V_prime_vertices)-1:
                angle_V_prime_vertices.append(slope_line(V_prime_vertices[i], V_prime_vertices[0]))
            else:    
                angle_V_prime_vertices.append(slope_line(V_prime_vertices[i], V_prime_vertices[i+1]))
                    
        i = 0 
        j = 0
        minkowski_sum = []  # Collecting vertices of the minkowski sum
        
        minkowski_sum.append((V_prime_vertices[i][0] + W_vertices[j][0], V_prime_vertices[i][1] + W_vertices[j][1]))
        
        while not((i==len(V_prime_vertices)-1) and (j==len(W_vertices)-1))  :
            
                if i == len(V_prime_vertices)-1:
                        j = j+1
                elif j == len(W_vertices)-1:
                        i = i+1
                elif slope_line(V_prime_vertices[i], V_prime_vertices[i+1]) < slope_line(W_vertices[j], W_vertices[j+1]) : #selecting the next vertex depending on angle transition
                        i = i+1
                elif slope_line(V_prime_vertices[i], V_prime_vertices[i+1]) > slope_line(W_vertices[j], W_vertices[j+1]): #selecting the next vertex depending on angle transition
                        j = j+1
                else:
                        i = i+1
                        j = j+1
                   
                minkowski_sum.append((V_prime_vertices[i][0] + W_vertices[j][0], V_prime_vertices[i][1] + W_vertices[j][1]))
                
        minkowski_sum_collection.append(sorted(set(minkowski_sum), key=minkowski_sum.index)) #collecting all the minkowski polygons

viridis = cm.get_cmap('viridis', len(thetha))
fig = plt.figure()
ax = fig.gca(projection='3d')

poly = PolyCollection(minkowski_sum_collection, facecolors = viridis.colors) #create a collection of polygons
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=thetha, zdir='z') # plotting polygons with thetha

ax.set_xlabel('X')
ax.set_xlim3d(-2.5, 2.5)
ax.set_ylabel('Y')
ax.set_ylim3d(0, 7.5)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 365)

plt.show()

