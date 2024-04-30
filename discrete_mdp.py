import numpy as np

def mat_policy_iteration(T, R, gamma, max_iters = 10):
    '''
    T is S x A x S
    R is S x A
    '''
    S = T.shape[0]
    A = T.shape[1]
    V = np.zeros((S, 1))
    Q = np.zeros((S, A))
    pi = np.zeros((S, A)) # S x A
    pi[:, 0] = 1

    policy_stable = False
    i = 0
    while not policy_stable and i < max_iters:     
        #print(pi, V)
        #print(V, "\n")
        
        if len(R.shape) > 2: 
            # Policy evaluation
            T_pi = np.vstack([pi[s, :].dot(T[s, :, :]) for s in range(S)]) # S x S
            R_T = np.array([[T[s, a, :].dot(R[s, a, :]) for a in range(A)] for s in range(S)])
            R_pi = np.hstack([R_T[s, :].dot(pi[s, :].T) for s in range(S)]) # S x 1
            V = np.linalg.inv(np.eye(S) - (gamma * T_pi)).dot(R_pi)

            # Policy improvement 
            policy_stable = True
            for s in range(S):
                Q[s, :] = [T[s, a, :].dot(R[s, a, :] + (gamma * V).flatten()) for a in range(A)]
                old_action = np.argmax(pi[s, :])
                new_action = np.argmax(Q[s, :])
                pi[s, :] = 0
                pi[s, new_action] = 1
                if old_action != new_action: 
                    policy_stable = False   
        else: 
            # Policy evaluation
            T_pi = np.vstack([pi[s, :].dot(T[s, :, :]) for s in range(S)]) # S x S
            R_pi = np.hstack([R[s, :].dot(pi[s, :].T) for s in range(S)]) # S x 1
            V = np.linalg.inv(np.eye(S) - (gamma * T_pi)).dot(R_pi)

            # Policy improvement 
            policy_stable = True
            for s in range(S): 
                Q[s, :] = [T[s, a, :].dot(R[s, a] + (gamma * V)) for a in range(A)]
                old_action = np.argmax(pi[s, :])
                new_action = np.argmax(Q[s, :])
                pi[s, :] = 0
                pi[s, new_action] = 1
                if old_action != new_action: 
                    policy_stable = False   
        i+=1
    return pi, Q, V.reshape(-1, 1)


def V_to_Q(V, T, R, gamma): 
    '''
    Given V (s,) returns Q (s x a)
    '''
    row_op = lambda T_r, R_r: T_r.dot(gamma * V + R_r)
    action_op = lambda T_a, R_a: [row_op(T_a[s, :], R_a[s]) for s in range(T.shape[0])] 

    if len(R.shape) > 2: 
        future_mat = np.array([action_op(T[:, a, :], R[:, a]) for a in range(T.shape[1])]).T
    else:
        future_mat = R + gamma * np.hstack([T[:, a, :].dot(V) for a in range(T.shape[1])])

    del row_op
    del action_op
    return future_mat 

def V_to_Q2(V, T, R, gamma): 
    '''
    Given V (s,) returns Q (s x a)
    '''
    if len(R.shape) > 2: 
        future_mat = np.array([[T[s, a, :].dot(gamma * V + R[s, a, :]) for a in range(T.shape[1])] for s in range(T.shape[0])])
    else:
        future_mat = R + gamma * np.hstack([T[:, a, :].dot(V) for a in range(T.shape[1])])

    return future_mat 


def value_iteration(T, R, gamma, delta = 0.1, verbose = False): 
    '''
    T = [S x A x S']
    R = [S x A x S']
    '''
    n_actions = T.shape[1]
    n_states = T.shape[0]
    V = np.zeros((n_states, ))
    Q = np.zeros((n_states, n_actions))
    max_change = delta
    i = 0
    while max_change >= delta: 
        max_change = 0.
        for s in range(Q.shape[0]):
            for a in range(n_actions): 
                Q[s, a] = np.dot(T[s, a, :], gamma * V + R[s, a, :])
                
            v = V[s]
            V[s] = np.amax(Q[s, :])
            max_change = max(max_change, np.abs(v - V[s]))
           

        if verbose: 
            print("iteration {}: max change {}".format(i, max_change))
        i+=1  

    actions = np.argmax(Q, axis = 1)
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states): 
        pi[s, actions[s]] = 1 

    return pi, Q, V



