import torch

def edge_information(topology: torch.Tensor):
    oriented_edges = torch.stack(
        (
            topology[:, [1, 2]],
            topology[:, [0, 2]],
            topology[:, [0, 1]],
        ),
        dim=1,
    ) 

    edge_reversed = oriented_edges[...,0] > oriented_edges[..., 1]
    canonical_edges = oriented_edges.sort(dim = -1).values

    edges, inverse = torch.unique(
        canonical_edges.reshape(-1, 2),
        dim=0,
        return_inverse=True,
    )

    element_to_edge = inverse.reshape(
        topology.shape[0], 3
    )

    return edges, element_to_edge, edge_reversed

def vectorized_local_to_global(
        topology: torch.Tensor,
        element_to_edge: torch.Tensor,
        edge_reversed: torch.Tensor,
        node_mapping: torch.Tensor,
        edge_offset: int,
        internal_offset: int,
        degree: int = 1
):

    num_triangles = topology.shape[0]
    num_local_DoFs = node_mapping.shape[0]
    mapping = torch.empty(
        (num_triangles, num_local_DoFs),
        dtype = torch.long
    )
    node_type = node_mapping % 10
    node_order = (node_mapping // 10) % 10 - 1
    edge_order = node_mapping // 100 - 1
    internal_order = node_mapping // 10 - 1
    vertex = torch.where(node_type == 1)[0]
    edge = torch.where(node_type == 2)[0]
    internal = torch.where(node_type == 3)[0]

    if torch.any(vertex):
        mapping[..., vertex] = topology[:, node_order[vertex]]
    if torch.any(edge):
        edge_mapping = element_to_edge[:, node_order[edge]] # global edge ID
        corrected_edge_order = torch.where(
            edge_reversed[:, node_order[edge]],
            degree - edge_order[edge] - 2,
            edge_order[edge])
        mapping[..., edge] = (edge_mapping*(degree - 1) + edge_offset + corrected_edge_order).to(torch.long)
    if torch.any(internal):
        mapping[..., internal] = ( torch.arange(num_triangles)[:, None] 
                                  * ((degree - 1) * (degree - 2)) // 2 +
                                  internal_offset 
                                  + internal_order[internal][None,:]
                                  )

    return mapping
        

def vectorized_edge_condition_to_dirichlet_mask(
        topology: torch.Tensor,
        boundary_conditions: torch.Tensor,
        element_to_edge: torch.Tensor,
        edge_offset: int,
        num_global_DoFs: int,
        degree: int = 1):
    dirichlet_mask = torch.zeros(
            (num_global_DoFs,),
            dtype = torch.bool
            )    
    dirichlet_edges = boundary_conditions == 1
    border_triangles = torch.any(dirichlet_edges, dim=-1)
    edge_indices = element_to_edge[dirichlet_edges]
    if degree > 1:
        edge_positions = torch.arange(
            degree - 1,
            device=topology.device,
        )

        edge_dofs = (
            edge_indices[:, None] * (degree - 1)
            + edge_offset
            + edge_positions[None, :]
        )

        dirichlet_mask[edge_dofs.reshape(-1)] = True
        selected_edges = dirichlet_edges[border_triangles]

    dirichlet_vertices = torch.stack(
        (
            selected_edges[:, 1] | selected_edges[:, 2],
            selected_edges[:, 0] | selected_edges[:, 2],
            selected_edges[:, 0] | selected_edges[:, 1],
        ),
        dim=-1,
    )

    selected_topology = topology[border_triangles]

    dirichlet_mask[
        selected_topology[dirichlet_vertices]
    ] = True

    return dirichlet_mask


def vectorized_local_1_1(
        geometry, 
        precomputed_gradhat_phihats, 
        quad_weights
        ):
    pts1 = geometry[..., 0, :]
    pts2 = geometry[..., 1, :]
    pts3 = geometry[..., 2, :]
    a = pts2[..., 0] - pts1[..., 0]
    b = pts3[..., 0] - pts1[..., 0]
    c = pts2[..., 1] - pts1[..., 1]
    d = pts3[..., 1] - pts1[..., 1]

    adjugate = torch.stack(
        (
            torch.stack((d, -b), dim=-1),
            torch.stack((-c, a), dim=-1),
        ),
        dim=-2,
    )
    determinant = a * d - b * c

    G = torch.einsum(
            "qia,eab->eqib",
            precomputed_gradhat_phihats,
            adjugate
            )
    A = torch.einsum(
            "q,eqia,eqja->eij",
            quad_weights,
            G,
            G,
            )

    return 1/torch.abs(determinant[...,None,None]) * A

def vectorized_global_1_1(
        topology: torch.Tensor,
        geometry: torch.Tensor,
        dirichlet_mask: torch.Tensor,
        global_mapping: torch.Tensor,
        precomputed_gradhat_phihats: torch.Tensor,
        quad_weights: torch.Tensor,
        ):
    
    A_local = vectorized_local_1_1(geometry[topology], precomputed_gradhat_phihats, quad_weights)


    num_triangles, num_local_dofs = global_mapping.shape

    rows = global_mapping[:, :, None].expand(
        num_triangles,
        num_local_dofs,
        num_local_dofs,
    )
    columns = global_mapping[:, None, :].expand(
        num_triangles,
        num_local_dofs,
        num_local_dofs,
    )

    local_dirichlet = dirichlet_mask[global_mapping]
    keep = (~local_dirichlet[:, :, None] & ~local_dirichlet[:, None, :])

    rows = rows[keep]
    columns = columns[keep]
    values = A_local[keep]
    boundary_indices = torch.where(dirichlet_mask)[0]

    rows = torch.cat(
        (rows, boundary_indices)
    )
    columns = torch.cat(
        (columns, boundary_indices)
    )

    values = torch.cat((values, torch.ones(boundary_indices.numel())))

    sparse_indices = torch.stack((rows, columns), dim=0)
    num_global_dofs = dirichlet_mask.numel()
    A = torch.sparse_coo_tensor(
        sparse_indices,
        values,
        size=(num_global_dofs, num_global_dofs),
    )

    return A.coalesce()

def f(x, y):
    return (
        2 * torch.pi**2
        * torch.sin(torch.pi * x)
        * torch.sin(torch.pi * y)
    )
def vectorized_local_0_f(
        geometry: torch.Tensor,
        precomputed_phihats: torch.Tensor,
        quad_pts: torch.Tensor,
        quad_weights: torch.Tensor,
        forcing
        ):
    pts1 = geometry[..., 0, :]
    pts2 = geometry[..., 1, :]
    pts3 = geometry[..., 2, :]
    a = pts2[..., 0] - pts1[..., 0]
    b = pts3[..., 0] - pts1[..., 0]
    c = pts2[..., 1] - pts1[..., 1]
    d = pts3[..., 1] - pts1[..., 1]

    jacobian = torch.stack(
        (
            torch.stack((a, b), dim=-1),
            torch.stack((c, d), dim=-1),
        ),
        dim=-2,
    )
    determinant = a * d - b * c
    # integrate
    pts = torch.einsum(
            "eij,qj->eqi",
            jacobian,
            quad_pts) + pts1[:,None,:]

    f_evaled = forcing(pts[...,-2], pts[..., -1])

    G = precomputed_phihats
    A = torch.einsum(
            "q,qi,eq->ei",
            quad_weights,
            G,
            f_evaled,
            )

    return A * torch.abs(determinant)[:, None]


def vectorized_global_0_f(
        topology: torch.Tensor,
        geometry: torch.Tensor,
        dirichlet_mask: torch.Tensor,
        global_mapping: torch.Tensor,
        precomputed_phihats: torch.Tensor,
        quad_pts: torch.Tensor,
        quad_weights: torch.Tensor,
        forcing
        ):
    
    A_local = vectorized_local_0_f(geometry[topology], precomputed_phihats, quad_pts, quad_weights, forcing)

    rows = global_mapping
    local_dirichlet = dirichlet_mask[global_mapping]
    keep = (~local_dirichlet)

    rows = rows[keep]
    values = A_local[keep]


    sparse_indices = torch.stack((rows,), dim=0)
    num_global_dofs = dirichlet_mask.numel()
    F = torch.sparse_coo_tensor(
        sparse_indices,
        values,
        size=(num_global_dofs,),
    )

    return F.coalesce()


