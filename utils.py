import numpy as np
import matplotlib.pyplot as plt 

def state_to_index(s, dim_sizes): 
    return np.ravel_multi_index(s, dim_sizes)

def index_to_state(ind, dim_sizes): 
    return np.unravel_index(ind, dim_sizes)  

