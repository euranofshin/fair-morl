import numpy as np
from .solvers.discrete_mdp import mat_policy_iteration
from .distributions import Uniform

# Action selection
def epsilon_greedy(s, Q, epsilon = 0.1): 
    n_actions = Q.shape[1]
    action_probs = np.ones(n_actions) * (epsilon/n_actions)
    max_index = np.where(Q[s, :] == np.amax(Q[s, :]))[0]
    action_probs[max_index] += (1 - epsilon)/max_index.shape[0]
    a = np.random.choice(n_actions, p = action_probs)
    return a


def greedy(s, Q):
    n_actions = Q.shape[1]
    action_probs = np.zeros(n_actions)
    max_index = np.where(Q[s, :] == np.amax(Q[s, :]))[0]
    action_probs[max_index] += 1./max_index.shape[0]
    a = np.random.choice(n_actions, p = action_probs)
    return a


def posterior_sampling(s, model, transition_func, R, gamma = 0.9): 
    '''
    model: samples parameters
    transition_func: takes sample and returns transition matrix
    '''
    sample = model.sample()
    T = transition_func(*sample)
    _, Q, _ = mat_policy_iteration(T, R, gamma)
    return greedy(s, Q)

'''
trans_func = lambda gamma, B: app_mdp.transitions_app(W = 5, 
                                G_user = 10, 
                                B_user = B, 
                                gamma_user = gamma, 
                                delta_B = -0.4, 
                                delta_gamma = 0.1, 
                                p_world = 0.8, 
                                d_world = 0.05)
R = app_mdp.rewards_app_goal_state(W = 5)
model = EnvironmentModel(Uniform(0, 0.99), Uniform(-5, 0))
print(posterior_sampling(0, model, trans_func, R))
'''
