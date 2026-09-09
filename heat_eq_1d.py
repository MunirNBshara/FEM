import torch
import matplotlib.pyplot as plt


L = 1
num_vertices = 101 

dx = L/(num_vertices-1)

# topological defines the vertices themselves in space
# geometric defines the vertices relationship with each other
topological_space = torch.arange(0, num_vertices) * dx
geometric_space = torch.stack((torch.arange(0, num_vertices-1), torch.arange(1, num_vertices)), dim=1,)
boundary_condition = torch.zeros(num_vertices-1, 2)

# opposite face dictates boundary condition
# 0, 1, 2 are none, dirichlet, neumann
boundary_condition[0, 1] = 1
boundary_condition[-1, 0] = 1

dirichlet_conditions = torch.zeros((2))
neumann_conditions = []


# du^2 / d^2 x = u_t
# v du^2 / d^2 x = v u_t
# int v du^2 / d^2x = int v u_t
# -int dv/dx du/dx + int d/dx ( v* du/dx) = int v u_t
# -int dv/dx du/dx + int_surface v * du/dx dot n = int v u_t
# v must satisfy 0 on boundary so:
# int dv/dx du/dx + int v u_t = 0
# Let v = sum phi_i v_i, let u = sum phi_i u_i
# We can write dv/dx = sum d phi_i / dx * v_i and same for u
# a_ij = d phi_i/dx * dphi_j/dx
# int v_i^T A u_j  + int v u_t = 0
# This must be true for all v(as in the span of the entire test basis
# In matrix for V A U + V M Udot = 0 -> A U + M Udot = 0

A = torch.zeros(num_vertices, num_vertices)
M = torch.zeros(num_vertices, num_vertices)
# Let phi_i be piecwise linear from 1 to 0 with any simplex it belongs to.
# So phi_j phi_i inner product is 0 unless they have a shared simplex
# (x - x_ip) / (x_i - x_ip) for previous segment. 
# (x_in - x_) / (x_in - x_i) for next segment
# derivatives:
# 1/(x_i - x_ip)
# 1/(x_i - x_in)
# multiplied and integral:
# x^2  

for simplex in geometric_space:
     dist = abs(topological_space[simplex[1]]-topological_space[simplex[0]])
     A[simplex[0], simplex[0]] += 1.0 / dist
     A[simplex[1], simplex[1]] += 1.0 / dist
     A[simplex[1], simplex[0]] = -1.0 / dist
     A[simplex[0], simplex[1]] = A[simplex[1], simplex[0]]

for simplex in geometric_space:
     dist = abs(topological_space[simplex[1]]-topological_space[simplex[0]])
     M[simplex[0], simplex[0]] += 2.0 * dist / 6
     M[simplex[1], simplex[1]] += 2.0 * dist / 6
     M[simplex[1], simplex[0]] = 1.0 * dist / 6
     M[simplex[0], simplex[1]] = 1.0 * dist / 6

u = torch.sin(torch.pi * topological_space)
print(u[0] + u[-1])
print(L)
print(topological_space)
dt = 0.0001
# M @ (U_new - U_old)/dt = - A U_old
eigenvalues = torch.linalg.eigvals(
    torch.linalg.solve(M, A)
).real

lambda_max = eigenvalues.max()

dt_stable = 2.0 / lambda_max
dt_nonoscillatory = 1.0 / lambda_max
print(dt_stable)
dt = dt_stable
for _ in range(10000):
    u = u - dt * torch.linalg.inv(M) @ A @ u
    u[0] = 0
    u[-1] = 0


plt.title(f"{dt*1000} secs")
plt.plot(topological_space, u)
plt.plot(topological_space, torch.exp(-torch.pi**2 * dt * 10000) * torch.sin(torch.pi * topological_space))
plt.show()





    


