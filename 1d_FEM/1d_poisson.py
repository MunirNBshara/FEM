
import torch
from fem_helpers_1d import (construct_global_1_1_operator_1d,
                            construct_global_0_0_operator_1d,
                            find_jacobian)
import matplotlib.pyplot as plt 
'''
Goal is to solve: u_xx = u_t - f
Integral form: 
    -integral u_x dot v_x + 
    integral_dirichlet v du/dn +
    integral_neumann v du/dn =
    integral v u_t - 
    integral v f
'''

vertices = 101
L = 1 
dx = L / (vertices-1)
topology = torch.stack((
        torch.arange(0, vertices-1),
        torch.arange(1, vertices)), dim=-1
        )
print(topology.shape)
geometry = torch.arange(0, vertices) * dx

# 0 = None, 1 = dirichlet, 2 = neumann
BCs = torch.zeros_like(topology)
BCs[0, 1] = 1
BCs[-1, 0] = 2

face_vertices = topology.flip(dims=(-1,))
dirichlet_vertices = torch.unique(face_vertices[BCs == 1])
neumann_vertices = torch.unique(face_vertices[BCs == 2])

free_mask = torch.ones(geometry.numel(), dtype=torch.bool)
free_mask[dirichlet_vertices] = False
free_vertices = torch.nonzero(free_mask, as_tuple=True)[0]



dirichlet_vals = torch.Tensor([0.0])
u_dir = torch.zeros_like(geometry)
u_dir.scatter_(0, dirichlet_vertices, dirichlet_vals)

J = find_jacobian(topology, geometry)
A = construct_global_1_1_operator_1d(J,topology)
M = construct_global_0_0_operator_1d(J,topology)
# AU + MUdot = integral_neumann v du/dn + integral v f
# A(u + u_dir) + M(u + u_dir)dot = integral_neumann v du/dn + integral v f

forcing_dirichlet = - A @ u_dir - M @ torch.zeros_like(geometry)
forcing_dirichlet = forcing_dirichlet[free_vertices]
forcing_neumann = torch.zeros_like(geometry)
neumann_vals = torch.Tensor([-1.0])
forcing_neumann.scatter_(0, neumann_vertices, neumann_vals)
forcing_neumann = forcing_neumann[free_vertices]

u = torch.sin(torch.pi*geometry[free_vertices])
forcing_function = torch.zeros_like(geometry) 
def func(x, t, L):
    center = L / 2 + 0.35 * L * torch.sin(
        torch.as_tensor(4 * torch.pi * t)
    )
    width = 0.035

    return 100.0 * torch.exp(
        -((x - center) / width)**2
    )


LHS = M[free_vertices][:, free_vertices]
dt = dx**2 / 6 
for i in range(10000):
    forcing_function_temp = torch.zeros_like(geometry) 
    forcing_function_temp[:-1] += dx / 2 * func(geometry[:-1] + dx/2, dt*i, L) 
    forcing_function_temp[1:] += dx / 2 * func(geometry[1:] - dx/2, dt*i, L)
    forcing_function = forcing_function_temp[free_vertices]
    RHS = - A[free_vertices][:, free_vertices] @ u + forcing_dirichlet + forcing_function + forcing_neumann
    u = torch.linalg.solve(LHS, LHS@ u  + dt * RHS)
    if(i%1000 == 0):
        plt.plot(geometry[free_vertices], u, label=f"iter={i}")
plt.legend()
plt.show()


