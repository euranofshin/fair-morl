import numpy as np
from utils import index_to_state, state_to_index
from discrete_mdp import mat_policy_iteration
import seaborn as sns
import matplotlib.pyplot as plt

A = 4
J = 2


def make_T(X = 6, Y = 4, reward_coord = [], p_north = 1.0, p_south =1.0, p_west = 1.0, p_east = 1.0, wall_coord = []): 
    S = X * Y
    dims = (X, Y)

    s_absorbing = state_to_index((X - 1, Y - 1), dims)
    s_walls = [state_to_index(coord, dims) for coord in wall_coord]
    print(s_walls)
    
    T_north = np.zeros((S, S)) 
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
             
            if y < Y - 1: 
                s_prime = state_to_index((x, y + 1), dims)
                if s_prime in s_walls: # wall 
                    T_north[s, s] = 1
                else: 
                    T_north[s, s_prime] = p_north
                    T_north[s, s] = 1 - p_north
            else: 
                T_north[s, s] = 1
    
    T_south = np.zeros((S, S)) 
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
            
            if y > 0: 
                s_prime = state_to_index((x, y-1), dims)
                if s_prime in s_walls: # wall 
                    T_south[s, s] = 1
                else: 
                    T_south[s, s_prime] = p_south
                    T_south[s, s] = 1 - p_south
            else:
                T_south[s, s] = 1
    
    T_east = np.zeros((S, S))
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)

            if x < X - 1: 
                s_prime = state_to_index((x + 1, y), dims)
                if s_prime in s_walls: # wall 
                    T_east[s, s] = 1
                else: 
                    T_east[s, s_prime] = p_east
                    T_east[s, s] = 1 - p_east
            else: 
                T_east[s, s] = 1 
    
    T_west = np.zeros((S, S))
    for x in range(X): 
        for y in range(Y):
            s = state_to_index((x, y), dims)
            if x > 0: 
                s_prime = state_to_index((x - 1, y), dims)
                if s_prime in s_walls: # wall 
                    T_west[s, s] = 1
                else: 
                    T_west[s, s_prime] = p_west
                    T_west[s, s] = 1 - p_west
            else: 
                T_west[s, s] = 1

    T = np.array([T_north, T_south, T_east, T_west])
    T = np.transpose(T, axes = [1, 0, 2]) # Make S A S dimensions


    # Add rewards as absorbing states
    for coord in reward_coord:
        s_reward = state_to_index(coord, dims)
        T[s_reward, :, :] = 0
        T[s_reward, :, s_reward] = 1

    
    #for a in range(4):
    #    print(T[:, a, :].sum(axis = 1))

    return T

def make_R(X = 5, Y = 4, reward_coord = [], rewards = []):
    '''
    w is the linear weight vector over rewards
    '''
    assert len(reward_coord) == len(rewards)

    S = X * Y
    dims = (X, Y)
    
    X = X - 1
    Y = Y - 1
    
    R_vec = np.zeros((S, A, S, J))
    np.array([np.zeros((S, A, S)) for j in range(J)])

    for reward, coord in zip(rewards, reward_coord): 
        s = state_to_index(coord, dims)
        R_vec[:, :, s, :] = np.tile(reward, S * A).reshape(S, A, J) # reward when entering
        R_vec[s, :, s, : ] = np.zeros((A, J)) # but only when entering from non-absorbing
    
    return R_vec

def plot_world(reward_coord = [], rewards = [], wall_coord = [], dims = (4, 3), scale = 1, traj_actions = [], traj_states = [], s0 = 0, colors = [], ax = None):
    X, Y = dims

    grid = np.zeros((X, Y)) 

    # Plot walls
    for coord in wall_coord: 
        grid[*coord] = 1
    
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=((X) * scale, (Y) * scale))
    grid = ax.imshow(grid.T, vmin = 0, vmax = 1, cmap = "Greys")
    
    # Plot rewards
    for coord, reward in zip(reward_coord, rewards):
        ax.text(coord[0] + (scale * 0.1) , coord[1] + (scale * 0.2), np.array2string(reward), size = 10 * scale, weight = 'bold')

    # Plot policy
    for color, actions, states in zip(colors, traj_actions, traj_states):
        coords = np.array([index_to_state(s, dims) for s in states])
        ax.plot(coords[:, 0], coords[:, 1], color = color, linewidth = 2)
        if coords.shape[0] > 1: 
            dx = (coords[-1, 0] - coords[-2, 0]) * 0.01
            dy = (coords[-1, 1] - coords[-2, 1]) * 0.01
        else: 
            dx = 0
            dy = 0.01
        ax.arrow(coords[-1,0], coords[-1, 1], dx, dy, width = 0.05, color = color)
    x0, y0 = index_to_state(s0, dims)
    plt.scatter(x0, y0, marker = '*', color = "black", s = scale * 100 , zorder = 10)


    ax.set_xticks(np.arange(-0.5, X, 1))
    ax.set_yticks(np.arange(-0.5, Y, 1))
    ax.grid(color='grey', linewidth=1)
        

    # Make things oriented right
    ax.invert_yaxis()



