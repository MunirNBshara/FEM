import torch
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from error_helpers import compute_eta, mark_refinement
from fem_helpers_2d import (
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

precomputed_gradhat_edge = gradhat_edge(quad_pts_1d, degree)
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

from amesh import AdaptiveMesh
from fespace import FESpace
from tle import TriangleLagrangeElement as TLE
from multigrid_helpers.multigrid_helpers import MG
from error_helpers import compute_eta 

ref_el = TLE(degree)

amesh = AdaptiveMesh(geometry, triangles, boundary_conditions)
amesh.build_view()
from AI_GENERATED_fem_graphing import plot_fem_mesh, plot_fem_value
show_plots = True
if show_plots:
    plot_fem_mesh(geometry, 
              triangles,
              degree)

l = 0
fe = FESpace(amesh.view(l), ref_el)
As, Fs = fe.create_matrices()
mg = MG(fe, As, [], 5, 5)
u = mg.mu_cycle(Fs.to_dense(), l, 1)
if show_plots:
    plot_fem_value(amesh.geometry, 
               amesh.cell_vertices[amesh.cell_active],
               u[fe.global_mapping],
               degree,
               show_mesh=False)
etas = compute_eta(fe, u)
marked_triangles = mark_refinement(fe, etas)
amesh.refinement_step(marked_triangles)

amesh.build_view()
if show_plots:
    plot_fem_mesh(amesh.geometry, 
              amesh.cell_vertices[amesh.cell_active],
              degree)

for l in range(1,35):
    fe = FESpace(amesh.view(l), ref_el)
    As, Fs = fe.create_matrices()
    fine_ids_maps, fine_to_curr_ids = amesh.one_level_embedding(l-1,l)
    mg.add_finer_subspace(fine_to_curr_ids, fe, As)
    u = mg.mu_cycle(Fs.to_dense(), l, 1)
    if show_plots:
        plot_fem_value(amesh.geometry, 
               amesh.cell_vertices[amesh.cell_active],
               u[fe.global_mapping],
               degree,
               show_mesh=False)

    etas = compute_eta(fe, u)
    marked_triangles = mark_refinement(fe, etas)
    print(f"{marked_triangles=}")
    print(f"{torch.all(amesh.cell_active[marked_triangles]==True)=}")
    print(f"{amesh.cell_children[marked_triangles]=}")
    amesh.refinement_step(marked_triangles)
    amesh.build_view()
    if show_plots:
        plot_fem_mesh(amesh.geometry, 
              amesh.cell_vertices[amesh.cell_active],
              degree)




