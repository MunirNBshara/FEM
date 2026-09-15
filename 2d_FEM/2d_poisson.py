import torch


from fem_helpers_2d import (
        global_0_f,
        gradhat_phihat,
        global_1_1,
        pts_bary_to_pts,
        ref_nodal_pt_to_identity_detailed,
        edge_condition_to_dirichlet_mask,
        phihat)
from fem_quadrature_2d import dunavant_rules

degree = 2
quad_bary_pts, quad_weights = dunavant_rules(degree)
quad_cart_pts = pts_bary_to_pts(quad_bary_pts)
precomputed_phihats = phihat(
        quad_cart_pts[..., -2],
        quad_cart_pts[..., -1],
        degree
        ) 
precomputed_gradhat_phihats = gradhat_phihat(
        quad_cart_pts[..., -2],
        quad_cart_pts[..., -1],
        degree)
simplex = torch.Tensor([[0, 0], [1, 0], [0, 1.0]])


cells_1d = 100
vertices_1d = cells_1d + 1

dx = 0.01
dy = 0.01
x_pts = torch.arange(vertices_1d, dtype=torch.int32) * dx
y_pts = torch.arange(vertices_1d, dtype=torch.int32) * dy

yy, xx = torch.meshgrid(
    y_pts,
    x_pts,
    indexing="ij",
)
geometry = torch.stack(
    [xx.flatten(), yy.flatten()],
    dim=-1,
).to(torch.float32)
triangles = torch.zeros((2*(cells_1d**2), 3), dtype = torch.int32)

boundary_conditions = torch.zeros_like(triangles)
for y_block in range(cells_1d):
    for x_block in range(cells_1d):
        # square coord (x_block, y_block) on grid
        triangle_idx = 2 * (y_block*cells_1d + x_block)
        triangles[triangle_idx] = torch.Tensor(
                [x_block + y_block*vertices_1d,
                 x_block+1 + y_block*vertices_1d,
                 x_block + (y_block+1)*vertices_1d])
        triangles[triangle_idx + 1] = torch.Tensor(
                [x_block+1 + (y_block+1)*vertices_1d, 
                 x_block + (y_block+1)*vertices_1d,
                 x_block+1 + y_block*vertices_1d])
        if y_block == 0:
            boundary_conditions[triangle_idx,2] = 1
        if x_block == 0:
            boundary_conditions[triangle_idx,1] = 1
        if y_block == cells_1d-1:
            boundary_conditions[triangle_idx+1,2] = 1
        if x_block == cells_1d-1:
            boundary_conditions[triangle_idx+1,1] = 1

top_to_edge = {}
edge_num = 0
for idx, triangle in enumerate(triangles):
    # identify edges
    edges = [triangle[1:], triangle[::2], triangle[:2]]

    for edge in edges:
        edge_key = (min(edge).item(), max(edge).item())
        if edge_key not in top_to_edge:
            top_to_edge[edge_key] = edge_num
            edge_num+=1
node_mapping = ref_nodal_pt_to_identity_detailed(degree)
   
number_of_edges = (
    3 * cells_1d**2
    + 2 * cells_1d
)

number_of_edge_points = (
    number_of_edges * (degree - 1)
)

from vectorized_fem_helpers_2d import (
        vectorized_local_to_global,
        edge_information,
        vectorized_edge_condition_to_dirichlet_mask,
        vectorized_global_1_1,
        vectorized_global_0_f)
from fem_helpers_2d import local_to_global
edges, element_to_edge, edge_reversed = edge_information(triangles)
edge_offset = vertices_1d**2
internal_offset = edge_offset + number_of_edge_points
num_global_DoFs = internal_offset + ((degree-1)*(degree-2) // 2) * triangles.shape[0]
global_mapping = vectorized_local_to_global(triangles,
        element_to_edge,
        edge_reversed,
        node_mapping,
        edge_offset,
        internal_offset,
        degree
)
dirichlet_mask = vectorized_edge_condition_to_dirichlet_mask(triangles,
        boundary_conditions,
        element_to_edge,
        vertices_1d**2,
        num_global_DoFs,
        degree)
A_sparse = vectorized_global_1_1(
        triangles,
        geometry,
        dirichlet_mask,
        global_mapping,
        precomputed_gradhat_phihats,
        quad_weights,
        )
F_sparse = vectorized_global_0_f(triangles,
        geometry,
        dirichlet_mask,
        global_mapping,
        precomputed_phihats,
        quad_cart_pts,
        quad_weights)

'''
import time

start = time.perf_counter()
dirichlet_mask = edge_condition_to_dirichlet_mask(
        triangles,
        boundary_conditions,
        top_to_edge,
        vertices_1d**2 + number_of_edge_points + ((degree-1)*(degree-2) // 2) * triangles.shape[0],
        vertices_1d**2,
        degree)
elapsed = time.perf_counter() - start
print("dirichlet_mask takes ", elapsed)

start = time.perf_counter()
A_sparse = global_1_1(triangles,
           geometry,
           dirichlet_mask,
           node_mapping,
           top_to_edge,
           vertices_1d**2,
           vertices_1d**2 + number_of_edge_points,
           precomputed_gradhat_phihats,
           quad_weights,
           degree
           )
elapsed = time.perf_counter() - start
print("Stiffness takes ", elapsed)

start = time.perf_counter()
F_sparse = global_0_f(triangles,
                      geometry,
                      dirichlet_mask,
                      node_mapping,
                      top_to_edge,
                      vertices_1d**2,
                      vertices_1d**2 + number_of_edge_points,
                      precomputed_phihats,
                        quad_cart_pts,
                      quad_weights,
                      degree)
elapsed = time.perf_counter() - start
'''
# print("RHS formation takes ", elapsed)
import scipy.sparse
import scipy.sparse.linalg

A_coo = A_sparse.coalesce().detach().cpu()

rows = A_coo.indices()[0].numpy()
cols = A_coo.indices()[1].numpy()
values = A_coo.values().numpy()

A_scipy = scipy.sparse.coo_matrix(
    (values, (rows, cols)),
    shape=A_coo.shape,
).tocsr()

b_numpy = F_sparse.to_dense().reshape(-1).detach().cpu().numpy()

u_numpy = scipy.sparse.linalg.spsolve(
    A_scipy,
    b_numpy,
)

u = torch.from_numpy(u_numpy)
print(u.shape)

import matplotlib.pyplot as plt
number_of_vertices = geometry.shape[0]

plt.scatter(
    geometry[:, 0],
    geometry[:, 1],
    c=u[:number_of_vertices],
    cmap="viridis",
)

plt.colorbar(label="u")
plt.axis("equal")
plt.show()
