import torch
def mark_refinement(V, etas: torch.Tensor, theta: float = 0.5):
    etas_squared = etas.square()
    eta_domain = torch.sqrt(etas_squared.sum())
    sorted_eta_squared, triangle_order = torch.sort(etas_squared, descending=True)
    cumulative_error = torch.cumsum(sorted_eta_squared, dim=0)

    infinum = theta * eta_domain
    number_marked = torch.searchsorted(cumulative_error, infinum).item() + 1 
    marked_triangles = triangle_order[:number_marked]
    

    return V.meshview.cell_ids[marked_triangles]

'''
Uses Dörfler metric to identify where refinement must be done:
    eta^2(u_T, M_T) >= theta eta^2(u_T, T)
where eta(u_T, M_T) = sqrt( sum_tau eta^2(u_T, tau))
and eta(u_T, tau) = sqrt( norm_(0, tau)[h (f + laplacian u_T)]^2 + sum_edges norm_(0, edge)[h^0.5 *(grad u_T dot normal_edge)]^2)
'''
def compute_eta(V, u: torch.Tensor):
    geometry = V.geometry
    quad_pts_tauhat = V.ref_element.triangle_quad_pts
    quad_weights_tauhat = V.ref_element.triangle_quad_weights
    quad_weights_edge = V.ref_element.edge_quad_weights
    global_mapping = V.global_mapping
    element_to_edge = V.meshview.element_to_edge
    forcing = V.forcing
    boundary_conditions = V.boundary_tags
    precomputed_hessianhat = V.ref_element.hessian_nodal_at_triangle_quad_pts
    precomputed_gradhat_edge = V.ref_element.grad_nodal_at_triangle_edge_quad_pts
    edge_elements = V.meshview.edge_count 
    
    
    vertices = geometry

    edge1_vector = vertices[..., 1, :] - vertices[..., 2, :]
    edge2_vector = -(vertices[..., 0, :] - vertices[..., 1, :])
    edge3_vector = -(vertices[..., 0, :] - vertices[..., 2, :])
    jac = torch.stack((edge2_vector, edge3_vector), dim = -1)
    edge1_length = torch.norm(edge1_vector, dim = -1)
    edge2_length = torch.norm(edge2_vector, dim = -1)
    edge3_length = torch.norm(edge3_vector, dim = -1)
    h_edge = torch.stack((edge1_length, edge2_length, edge3_length), dim = -1)
    h_tau = torch.max(h_edge, dim = -1)[0]

    # integrate (f + laplacian u_T) over tau
    quad_pts_tau = torch.einsum("eij,qj->eqi",
                                   jac, quad_pts_tauhat) + vertices[:, None, 0, :]

    evaled_pts = forcing(quad_pts_tau[..., 0], quad_pts_tau[..., 1])
    invjacs = torch.linalg.inv(jac)
    detjacs = torch.abs(torch.linalg.det(jac))
    local_basis = u[global_mapping]
    laplacian = torch.einsum("qnba,ebd,ead,en->eq",
        precomputed_hessianhat,
        invjacs,
        invjacs,
        local_basis)
    G = evaled_pts + laplacian
    tau_integral = torch.einsum("eq,eq,q->e",
                 G,
                 G,
                 quad_weights_tauhat) * h_tau**2 * detjacs 

    edge1_hat_normal = torch.Tensor([1.0, 1.0] )/ 2.0**(0.5)
    edge2_hat_normal = torch.Tensor([-1.0, 0.0]) 
    edge3_hat_normal = torch.Tensor([0.0, -1.0])
    hat_normals = torch.stack((edge1_hat_normal, edge2_hat_normal, edge3_hat_normal), dim = -1)
    normals = torch.einsum("bl,ebd->eld",
                           hat_normals,
                           invjacs 
                           )
    unit_normals = normals / torch.linalg.vector_norm(normals, dim = -1, keepdim=True)

    grad_u = torch.einsum(
        "qnal,ead,en->elqd",
        precomputed_gradhat_edge,
        invjacs,
        local_basis,
    )
    normal_flux = torch.einsum(
        "elqd,eld->elq",
        grad_u,
        unit_normals
    )
    # Now combine fluxes meaning map each el to an edge

    edge_sums = torch.zeros((edge_elements, quad_weights_edge.shape[0]), dtype=torch.float32)
    

    edge_sums.index_add_(
        0,
        element_to_edge.reshape(-1),
        normal_flux.reshape(
            -1,
            quad_weights_edge.shape[0],
        )
    )
    dirichlet_edge_ids = element_to_edge[
        boundary_conditions == 1
    ]

    edge_sums[dirichlet_edge_ids] = 0
    edge_integrals = torch.einsum(
        "gq,gq,q->g",
        edge_sums,
        edge_sums,
        quad_weights_edge,
    )
    element_edge_integrals = edge_integrals[element_to_edge]
    element_edge_error = torch.einsum("el,el->e",
                                      element_edge_integrals,
                                      h_edge**2)
    return torch.sqrt(element_edge_error + tau_integral)



