from gridworld import * 
from chvi import * 

# Example from paper
'''
X = 5
Y = 3
S = X * Y
dims = (X, Y)
A = 4
wall_coord = [(1,0), (2,0), (3,0), (1,2), (2,2), (3,2)]
reward_coord = [(0, 0), (0, Y - 1), (X - 1, 0), (X - 1, Y - 1)]
rewards = np.array([[0.7, 0.4], [0.6, 0.6], [0, 1], [1, 0]])
gamma = 0.9
s0 = state_to_index((2, 1), dims)

T = make_T(wall_coord = wall_coord, X = X, Y = Y, reward_coord = reward_coord)
R = make_R(X = X, Y = Y, reward_coord = reward_coord, rewards = rewards)

print("BEFORE CODE")
s = s0
a = 0
S_ = np.where(T[s, a, :] > 0)[0]
for s_ in S_:
    print(s, a, s_, ":", T[s,a,s_])
print()


Q = chvi(T, R, gamma) 
policies, states, colors = plot_reward_space(Q, T, s0 = s0, by_policy = True)
print(policies)
print(states)

plot_world(reward_coord = reward_coord, rewards = rewards, wall_coord = wall_coord, dims = dims, scale = 1, 
        policies = policies,  states = states, s0 = s0, colors = colors)
'''

X = 4
Y = 4
S = X * Y
dims = (X, Y)
A = 4
reward_coord = [(0, Y - 1), (X - 1, 0)]
rewards = np.array([[0, 1], [1, 0]])

# Example of subgroup 1
wall_coord = []
gamma = 0.9
s0 = state_to_index((0, 0), dims)

T = make_T(wall_coord = wall_coord, X = X, Y = Y, reward_coord = reward_coord)
R = make_R(X = X, Y = Y, reward_coord = reward_coord, rewards = rewards)

Q = chvi(T, R, gamma) 

plot_world(reward_coord = reward_coord, rewards = rewards, wall_coord = wall_coord, dims = dims, scale = 1,
       s0=s0)
plt.show()
plot_reward_space(Q, T, s0 = s0, by_policy = False)
# Example of subgroup 2
wall_coord = [(1, 0), (1, 1)]
gamma = 0.9
s0 = state_to_index((0, 0), dims)

T = make_T(wall_coord = wall_coord, X = X, Y = Y, reward_coord = reward_coord)
R = make_R(X = X, Y = Y, reward_coord = reward_coord, rewards = rewards)

Q = chvi(T, R, gamma) 
plot_reward_space(Q, T, s0 = s0, by_policy = False)

