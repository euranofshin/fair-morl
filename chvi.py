# Convex hull value iteration
import numpy as np
from scipy.spatial import ConvexHull
from utils import state_to_index, index_to_state
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

def hull_addition(Q, U): 
    '''
    Eq. 4 of paper
    '''
    sum_set = np.array([q + u for q in Q for u in U])
    unique, unique_indices  = np.unique(sum_set, return_index = True, axis = 0)
    sum_set = sum_set[unique_indices, :]
    if sum_set.shape[0] > 2: 
        vertices = ConvexHull(sum_set).vertices
        return sum_set[vertices, :]
    else: 
        return sum_set
    
def Q_star(Q, w_vec, S, A):
    Q_ = np.zeros((S, A))
    w_vec = w_vec.reshape(-1, 1)
    for s in range(S): 
        for a in range(A): 
            Q_sa = Q[(s, a)] # N x J, J x 1
            Q_[s, a] = np.max(Q_sa.dot(w_vec))
    return Q_

def check_convergence(Q_old, Q_new, S, A):
    # Make sure shapes are same
    shapes_match = True
    for s in range(S): 
        for a in range(A): 
            if Q_old[(s, a)].shape != Q_new[(s, a)].shape:
                shapes_match = False
    
    if shapes_match: # Find max difference in values
        max_delta = 0
        for s in range(S): 
            for a in range(A): 
                delta = np.abs(Q_old[(s, a)] - Q_new[(s, a)]).max()
                if  delta > max_delta: 
                    max_delta = delta
        print("Shapes match, delta is {}...".format(max_delta))
        return max_delta
    else: 
        print("Not converged because shapes don't match...")
        return np.inf

import matplotlib.pyplot as plt

def chvi(T, R, gamma, max_delta = 0.1):
    '''
    R is reward matrix of shape [S, A, J]
    T is transition matrix of shape [S, A, S]
    '''
    S, A, _, J = R.shape

    # Initialize Q
    Q  = {}
    for s in range(S): 
        for a in range(A): 
            Q[(s, a)] = np.random.uniform(0,1, size = (1, J))
            #Q[(s, a)] = np.zeros((1, J))

    converged = False
    i = 0
    while not converged: 
        # Do the update
        Q_old = deepcopy(Q)
        for s in range(S): 
            for a in range(A): 
                Q_sa = np.zeros((1, J))
                S_ = np.where(T[s, a, :] > 0)[0] # Consider only non-zero transitions for speed
                for s_ in S_:
                    U_action_hulls = np.vstack([Q[(s_, a_)] for a_ in range(A)])
                    unique, unique_indices  = np.unique(U_action_hulls, return_index = True, axis = 0)
                    U_action_hulls = U_action_hulls[unique_indices, :]
                    if U_action_hulls.shape[0] > 2: 
                        vertices = ConvexHull(U_action_hulls).vertices
                        hull = U_action_hulls[vertices, :]
                    else: 
                        hull = U_action_hulls
                    #print("Input num {} output num {}".format(U_action_hulls.shape[0], hull.shape[0]))
                    linear_Q = R[s, a, s_, :] + gamma * hull
                    p_linear_Q = T[s, a, s_] * linear_Q
                    Q_sa = hull_addition(Q_sa, p_linear_Q)
                Q[(s, a)] = Q_sa
            #if(i > 2 and s == 0):
            #    plt.scatter(U_action_hulls[:, 0], U_action_hulls[:, 1])
            #    plt.scatter(hull[:, 0], hull[:, 1], color = "red")
            #    plt.show()
        
    
        # Check for convergence
        delta = check_convergence(Q_old, Q, S, A)
        converged = (delta < max_delta)
        i+=1
    return Q

def follow_trajectory(s0, pi, T):
    '''
    For a deterministic policy. Follows most likely trajectory.
    Returns the trajectory of actions, states starting from state s0
    '''
    states = [s0]
    actions = []
    repeat = False
    s = s0
    
    while not repeat: 
        a = pi[s]
        s_ = T[s, a, :].argmax()
        actions.append(a)
        states.append(s_)
        # Check if repeat (abs state)
        repeat = (s_ == s)
        s = s_
    return states[:-1], actions # leave off last state because it repeated


def plot_reward_space(Q, T, N1 = 10, N2 = 10, s0 = 1, by_policy = False, ax = None, vmin = 0, vmax = 1, cmap = None): 
    # Grid of reward values
    X = np.linspace(0.1, 1, N1)
    Y = np.linspace(0.1, 1, N2)
    xz, yz = np.meshgrid(X, Y)
    zz = np.empty(xz.shape, dtype="object")
    ss = np.empty(xz.shape, dtype = "object")
    S, A,_ = T.shape
    action_mapping = np.array(["^", "v", ">", "<"])
    policies = np.zeros((N1, N2, S))
    qq = np.zeros((N1, N2))
    for i in range(N1): 
        for j in range(N2): 
            # Find policy
            x = xz[i, j]
            y = yz[i, j]
            Q_ = Q_star(Q, np.array([x, y]), S, A)
            pi = Q_.argmax(axis = 1).astype(int)
            traj_states, traj_actions = follow_trajectory(s0, pi, T)

            # Save the actions from the start state
            zz[i, j] = " ".join(list(action_mapping[traj_actions]))
            # Save the states from the state state
            ss[i, j] = " ".join(map(str, traj_states))
            # Save the optimal policies
            policies[i, j, :] = pi
            # Save the Q values at start state
            qq[i, j] = Q_[s0, pi[s0]]
     
    # Enumerate unique policies from start state
    unique_policies, color_index, policy_indices = np.unique(zz.flatten(), return_inverse = True, return_index = True)
    U = unique_policies.shape[0]
    plot_policies = policy_indices.reshape((N1, N2))
    
    
    if by_policy: 
        if ax is None: 
            fig, ax = plt.subplots(1,1, figsize=(7.5, 4))
        # Plot policies
        df = pd.DataFrame(plot_policies, columns = np.around(Y, 2), index = np.around(X, 2))
        cmap = "tab10" if cmap is None else cmap
        sns.heatmap(df, cbar = False, cmap = cmap, vmin = 0, vmax = U)
        ax.invert_yaxis()

        # Add legend for policies
        colors = ax.collections[0].get_facecolors()
        #legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors[color_index], zz.flatten()[color_index])]
        #plt.legend(handles=legend_patches, title='Legend', loc='upper right', bbox_to_anchor = (1.5,1))
    else: 
        if ax is None: 
            fig, ax = plt.subplots(1,1)
        colors = np.zeros(policy_indices.shape)
        df = pd.DataFrame(qq, columns = np.around(Y,2), index = np.around(X, 2))
        sns.heatmap(df, cbar = False, cmap = "Greens", vmin = vmin, vmax = vmax)
        ax.invert_yaxis()

    #plt.tight_layout()
    #plt.savefig("test.pdf", bbox_inches = "tight")
    #plt.show()
    
    unique_policy_states = ss.flatten()[color_index]
    unique_policies = [pi.split(" ") for pi in unique_policies]
    unique_policy_states = [[int(s) for s in t.split(" ")] for t in unique_policy_states]

    
    # Only return unique policies
    policies_to_return = np.unique(policies.reshape(-1, S), axis = 0)
    return unique_policies, unique_policy_states, colors[color_index], policies_to_return, qq 




    

def vi(T, R, gamma):
    S, A, _, J = R.shape
    Q = np.zeros((S, A))
    V = np.zeros((S, ))
    i = 0 
    while i < 5: 
        for s in range(S):  
            for a in range(A): 
                S_ = np.where(T[s, a, :] > 0)[0] # Consider only non-zero transitions for speed
                for s_ in S_:
                    Q[s, a] = (T[s, a, s_] * (R[s, a, s_, 0] + (gamma * V[s_])))
            a_max = np.argmax(Q[s, :])
            V[s] = Q[s, a_max]



        for y in reversed(range(3)):
            string = []
            for x in range(3): 
                s = state_to_index([x, y], (3, 3))
                string.append(np.around(Q[s, a], 1))
                #string.append(R[:, 0, s, 0].max())
            print(string)
        print("\n")

        i+=1
    return Q

