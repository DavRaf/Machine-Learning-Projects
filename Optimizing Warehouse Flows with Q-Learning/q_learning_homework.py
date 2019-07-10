# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:02:20 2019

@author: David
"""

# importing the libraries
import numpy as np

# setting the parameters gamma and alpha for the Q-Learning
gamma = 0.75
alpha = 0.9 # learning rate

# PART 1 - DEFINING THE ENVIRONMENT

# defining the states
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# definining the actions
actions = [i for i in range(0, 12)]

# defining the rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,0,0,1,0,0,0,0,0,0],
            [0,1,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0],
            [0,1,0,0,0,0,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,1,0,0,0,0],
            [0,0,0,1,0,0,1,0,0,0,0,1],
            [0,0,0,0,1,0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0,0,1,0,1,0], # increase (till 500) the reward going from J to K, or decrease (till -500) the reward going from J to F
            [0,0,0,0,0,0,0,0,0,1,0,1],
            [0,0,0,0,0,0,0,1,0,0,1,0]])

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

# making the final function that will return the optimal route
def route(starting_location, ending_location):
    R_new = np.copy(R) # avoid R_new = R
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# PART 3 - GOING INTO PRODUCTION

def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

# printing the final route
route = best_route('E', 'K', 'G')
print('Route:'.format(), route)
        