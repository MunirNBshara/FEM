import torch
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from adaptive_helpers import compute_eta, mark_refinement
from fem_helpers_2d import (
        global_0_f,
        gradhat_phihat,
        global_1_1,
        pts_bary_to_pts,
        ref_nodal_pt_to_identity_detailed,
        edge_condition_to_dirichlet_mask,
        phihat,
        gradhat_edge,
        hessianhat)
from fem_quadrature_2d import (dunavant_rules, gaussian_quadrature)

degree = 2
quad_bary_pts, quad_weights = dunavant_rules(degree)
quad_pts_1d, quad_weights_1d = gaussian_quadrature(degree)
quad_cart_pts = pts_bary_to_pts(quad_bary_pts)

precomputed_gradhat = gradhat_edge(quad_pts_1d, degree)
precomputed_hessianhat = hessianhat(
        quad_cart_pts[..., -2],
        quad_cart_pts[..., -1],
        degree)
precomputed_phihats = phihat(
        quad_cart_pts[..., -2],
        quad_cart_pts[..., -1],
        degree) 
precomputed_gradhat_phihats = gradhat_phihat(
        quad_cart_pts[..., -2],
        quad_cart_pts[..., -1],
        degree)

simplex = torch.Tensor([[0, 0], [1, 0], [0, 1.0]])
cells_1d = 2 
vertices_1d = cells_1d + 1
dx = dy = 0.5

x_pts = torch.arange(vertices_1d, dtype=torch.float32) * dx
y_pts = torch.arange(vertices_1d, dtype=torch.float32) * dy

yy, xx = torch.meshgrid(y_pts, x_pts, indexing="ij")
geometry = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

triangle_list = []
cut = cells_1d // 2

for y_block in range(cells_1d):
    for x_block in range(cells_1d):
        if x_block >= cut and y_block >= cut:
            continue

        bottom_left = x_block + y_block * vertices_1d
        bottom_right = bottom_left + 1
        top_left = bottom_left + vertices_1d
        top_right = top_left + 1

        triangle_list.extend([
            [bottom_left, bottom_right, top_left],
            [top_right, top_left, bottom_right]
        ])

triangles = torch.tensor(triangle_list, dtype=torch.long)
used_vertices, inverse = torch.unique(
    triangles.flatten(), sorted=True, return_inverse=True
)
geometry = geometry[used_vertices]
triangles = inverse.reshape(-1, 3)
local_edges = [(1, 2), (0, 2), (0, 1)]

top_to_edge = {}
edge_incidence = {}

for triangle_idx, triangle in enumerate(triangles.tolist()):
    for local_edge_idx, (a, b) in enumerate(local_edges):
        edge_key = tuple(sorted((triangle[a], triangle[b])))

        if edge_key not in top_to_edge:
            top_to_edge[edge_key] = len(top_to_edge)
            edge_incidence[edge_key] = []

        edge_incidence[edge_key].append(
            (triangle_idx, local_edge_idx)
        )

boundary_conditions = torch.zeros_like(triangles)

for incident_triangles in edge_incidence.values():
    if len(incident_triangles) == 1:
        triangle_idx, local_edge_idx = incident_triangles[0]
        boundary_conditions[triangle_idx, local_edge_idx] = 1

node_mapping = ref_nodal_pt_to_identity_detailed(degree)
number_of_edges = len(top_to_edge)
number_of_edge_points = number_of_edges * (degree - 1)

def forcing(x, y):
    return (
        2 * torch.pi**2
        * torch.sin(torch.pi * x)
        * torch.sin(torch.pi * y)
    )

from adaptive_helpers import AdaptiveMesh
from vectorized_fem_helpers_2d import (
        vectorized_local_to_global,
        edge_information,
        vectorized_edge_condition_to_dirichlet_mask,
        vectorized_global_1_1,
        vectorized_global_0_f)

amesh = AdaptiveMesh(geometry, triangles, boundary_conditions)
amesh.build_neighbors()
max_eta = 10
while max_eta > 0.02:
    print(max_eta)
    active = amesh.cell_active
    geometry = amesh.geometry
    topology = amesh.cell_vertices[active]
    boundary_conditions = amesh.cell_dirichlet[active]
    edges, element_to_edge, edge_reversed = edge_information(topology)
    edge_offset = geometry.shape[0] 
    internal_offset = edge_offset + edges.shape[0]*(degree-1)
    num_global_DoFs = internal_offset + ((degree-1)*(degree-2) // 2) * topology.shape[0]
    global_mapping = vectorized_local_to_global(topology,
        element_to_edge,
        edge_reversed,
        node_mapping,
        edge_offset,
        internal_offset,
        degree
    )
    dirichlet_mask = vectorized_edge_condition_to_dirichlet_mask(topology,
        boundary_conditions,
        element_to_edge,
        edge_offset,
        num_global_DoFs,
        degree)
    A_sparse = vectorized_global_1_1(
        topology,
        geometry,
        dirichlet_mask,
        global_mapping,
        precomputed_gradhat_phihats,
        quad_weights
        )
    F_sparse = vectorized_global_0_f(topology,
        geometry,
        dirichlet_mask,
        global_mapping,
        precomputed_phihats,
        quad_cart_pts,
        quad_weights,
        forcing)

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
    
    from AI_GENERATED_fem_graphing import plot_fem_mesh, plot_fem_value
    plot_fem_mesh(geometry, 
                  topology,
                  degree)
#   plt.scatter(geometry[:,0],
#               geometry[:,1],
#               c=u[:geometry.shape[0]],
#               cmap="viridis",
#               )
#   plt.colorbar(label="u")
#   plt.axis("equal")
#   plt.show()
    plot_fem_value(geometry, topology, u[global_mapping], degree, show_mesh=False)
    
    etas = compute_eta(topology,
                       geometry,
                       quad_cart_pts,
                       quad_weights,
                       quad_weights_1d,
                       global_mapping,
                       u,
                       element_to_edge,
                       forcing,
                       boundary_conditions,
                       precomputed_hessianhat,
                       precomputed_gradhat,
                       internal_offset - edge_offset)
    max_eta = torch.sqrt(torch.sum(etas**2))

    marked_triangles = mark_refinement(etas, 0.4)
    cells = torch.arange(amesh.cell_count) 
    amesh.refinement_step(cells[active][marked_triangles])
    amesh.build_neighbors() 
