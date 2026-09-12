import torch
import math


'''
Generates polynomial in least degree to highest degree nested with highest degree x to highest degree y.
Ex degree 2:
    1,x,y,x^2,xy,y^2
'''
def ref_pt_to_poly(x: torch.Tensor, y: torch.Tensor, degree: int = 1):
    poly = []
    for i in range(degree+1):
        for j in range(i+1):
            poly.append(x**(i-j) * y**(j))
    return torch.stack(poly, dim=-1)
'''
Generates Nd x 2 matrix representing gradient of the polynomial expansion
'''
def ref_pt_to_poly_gradient(x: torch.Tensor, y:torch.Tensor, degree: int = 1):
    terms = (degree+1) * (degree + 2) // 2
    poly_grad_x = torch.zeros(x.shape[-1], terms, dtype = x.dtype)
    poly_grad_y = torch.zeros(y.shape[-1], terms, dtype = x.dtype)
    idx = 0
    for i in range(degree+1):
        for j in range(i+1):
            if j!=0:
                poly_grad_y[..., idx] = (x**(i-j) * j * y**(j-1))
            if (i-j)!=0:
                poly_grad_x[..., idx] = ((i-j)*x**(i-j-1) * y**(j))
            idx+=1
    return torch.stack((poly_grad_x,poly_grad_y), dim=-1)



'''
Generates appopriate points on reference triangular simplex for an nth degree polynomial
Generation is based on barycentric coordinates where
i/deg + k/deg + j/deg = 1

Ordering goes in increasing L1 norm with equivalent norm being ordered by ascending y
'''
def ref_nodes_cart(degree: int = 1):
    nodes = []

    for i in range(degree, -1, -1):
        for j in range(degree-i,-1,-1):
            k = degree - i - j
            nodes.append((j / degree, k / degree))

    return torch.tensor(nodes)

def ref_nodes_bary(degree: int = 1):
    nodes = []

    for i in range(degree, -1, -1):
        for j in range(degree-i,-1,-1):
            k = degree - i - j
            nodes.append((1 - i / degree, j / degree, k / degree))

    return torch.tensor(nodes)
'''
1 vertex,
2 edge,
3 internal
'''
def node_identifier(bary_pts: torch.Tensor, tol = 1e-6):
    bary_pts_type = torch.zeros(bary_pts.shape[-2])
    for idx in range(bary_pts.shape[-2]):
        pt = bary_pts[..., idx, :]
        nz_mask = torch.where(abs(pt)>tol, 1.0,0.0)
        nnz = torch.sum(nz_mask, dim =-1)
        if nnz < 1:
            bary_pts_type[idx] = 3
        elif nnz < 2:
            bary_pts_type[idx] = 2
        else:
            bary_pts_type[idx] = 1
    return bary_pts_type

'''
units digit:
1 vertex,
2 edge,
3 internal

tens digit:
vertex/edge number

hundreds+:
relative order
'''
def detailed_node_identifier(degree = 1):
    identifiers = []
    internal_ct = 1
    for i in range(degree, -1, -1):
        for j in range(degree-i,-1,-1):
            k = degree - i - j
            nz = int(i > 0) + int(k > 0) + int(j > 0)
            identifier = nz # units digit
            if identifier == 1:
                if i==0:
                    if j==0:
                        identifier+=3*10
                    elif k==0:
                        identifier+=2*10
                elif j==0:
                    if k==0:
                        identifier+=1*10

            elif identifier == 2:
                if i==0:
                    identifier+=10 + 100*k 
                elif j==0:
                    identifier+=20 + 100*k
                elif k==0:
                    identifier+=30 + 100*j
            elif identifier == 3:
                identifier+=10*internal_ct
                internal_ct+=1
            identifiers.append(identifier)
    return torch.tensor(identifiers)


'''
Finds appropriate polynomial functions for delta_mn linear function on every node
each row is 1 polynomial function of appropriate degree that ensures at the everything
except the reference node at interest is 0.

evaluated polynomial -> nodal coordinates
'''
def poly_to_nodal_coords(degree: int = 1):
    nodes = ref_nodes_cart(degree)
    xhat = nodes[:, 0]
    yhat = nodes[:, 1]

    V = ref_pt_to_poly(xhat, yhat, degree).T
    C = torch.linalg.solve(V, torch.eye(
        V.shape[0],
    ))

    return torch.where(C.abs() > 1e-6, C, torch.zeros_like(C))


def cartesian_index(x_power: int, y_power: int) -> int:
    total = x_power + y_power
    return total * (total + 1) // 2 + y_power

def barycentric_index(
    i: int,
    j: int,
    k: int,
    degree: int,
) -> int:
    assert i + j + k == degree

    block = degree - i
    return block * (block + 1) // 2 + k

'''
evaluated polynomial to barycentric polynomial
'''
def poly_to_bpoly(degree=1):
    # lambda_1 = (1 -xhat - yhat)
    # lambda_2 = xhat
    # lambda_3 = yhat

    # change of basis deg 1
    terms = (degree + 1) * (degree + 2) // 2

    A = torch.zeros((terms, terms))

    row = 0
    for i in range(degree, -1, -1):
        for j in range(degree - i, -1, -1):
            k = degree - i - j

            # computation of cross terms from lambda_1
            for x_degree_lambda1 in range(i+1):
                for y_degree_lambda1 in range(i+1-x_degree_lambda1):

                    # how many ways from i can we create x degree, 
                    # how many ways can we make y degree
                    # rest is just 1's
                    coefficient = (
                        (-1) ** (x_degree_lambda1 + y_degree_lambda1)
                        * math.comb(i, x_degree_lambda1)
                        * math.comb(i - x_degree_lambda1, y_degree_lambda1)
                    )

                    x_deg = x_degree_lambda1 + j
                    y_deg = y_degree_lambda1 + k
                    column = cartesian_index(
                        x_deg,
                        y_deg,
                    )
                    A[row, column] = coefficient
            row+=1

    return A

'''
barycentric polynomial to barycentric coords
'''
def bpoly_to_bcoords(bary: torch.Tensor, degree: int):
    lambda1_d = bary[..., 0]
    lambda2_d = bary[..., degree * (degree + 1) // 2]
    lambda3_d = bary[..., -1]

    pure_powers = torch.stack(
        [lambda1_d, lambda2_d, lambda3_d],
        dim=-1,
    )

    return pure_powers.clamp_min(0).pow(1.0 / degree)

'''
barycentric coords to barycentric polynomial
'''
def bary_coords_to_poly(coords: torch.Tensor, degree: int):
    lambda1 = coords[...,0]
    lambda2 = coords[...,1]
    lambda3 = coords[...,2]
    poly = []
    for i in range(degree, -1, -1):
        for j in range(degree-i, -1, -1):
            k = degree - j - i 
            poly.append(lambda1**i * lambda2**j * lambda3**k)
    return torch.stack(poly, dim=-1)

'''
barycentric polynomial to nodal coords
'''
def bary_poly_to_nodal(poly_b: torch.Tensor, degree: int):
    A = poly_to_bpoly(degree)
    C = poly_to_nodal_coords(degree)

    poly_cartesian = torch.linalg.solve(A, poly_b)
    return C @ poly_cartesian

def bary_to_cart(bary_pts: torch.Tensor):
    return torch.stack((bary_pts[...,-2],bary_pts[...,-1]), dim=-1)
'''
Takes simplex, outputs local stiffness matrix
'''
def local_1_1(simplex: torch.Tensor,
              precomputed_gradhat_phihats: torch.Tensor,
              weights: torch.Tensor):
    x1, y1 = simplex[0]
    x2, y2 = simplex[1]
    x3, y3 = simplex[2]

    # must be computed every local index, can replace with adjoint
    dx_dxhat = torch.Tensor([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
    dxhat_dx = torch.linalg.inv(dx_dxhat)
    scaling = torch.abs(torch.linalg.det(dx_dxhat))

    # integrate

    quadrature_size, nodal_basis_size = precomputed_gradhat_phihats.shape[:2]
    A = torch.zeros((nodal_basis_size, nodal_basis_size), dtype=torch.float32)
    G = precomputed_gradhat_phihats @ dxhat_dx

    A = torch.einsum(
        "q,qia,qja->ij",
        weights,
        G,
        G,
    )

    return scaling * A

def global_1_1(topology: torch.Tensor,
               geometry: torch.Tensor,
               dirichlet_mask: torch.Tensor,
               node_mapping: torch.Tensor,
               top_to_edge: dict,
               edge_offset: int,
               internal_offset: int,
               precomputed_gradhat_phihats: torch.Tensor,
               weights: torch.Tensor,
               degree: int = 1
               ):
    num_internal_pts = (degree-1)*(degree-2) // 2
    free_nodal_points =  int((~dirichlet_mask).sum().item())
    nodal_points = internal_offset + num_internal_pts*topology.shape[0]
    row_parts = []
    col_parts = []
    value_parts = []
    boundary_indices = torch.nonzero(
        dirichlet_mask,
        as_tuple=False,
    ).flatten()

    for idx, points in enumerate(topology):
        simplex = torch.stack((geometry[points[0]], geometry[points[1]], geometry[points[2]]),
                              dim=0)
        A_local = local_1_1(simplex, precomputed_gradhat_phihats, weights) 
        local_to_global_mapping = local_to_global(points,
                        idx,
                        node_mapping,
                        top_to_edge,
                        edge_offset,
                        internal_offset,
                        degree
                        )
        n = local_to_global_mapping.numel()

        rows = local_to_global_mapping[:, None].expand(n, n)
        cols = local_to_global_mapping[None, :].expand(n, n)
        local_free = dirichlet_mask[local_to_global_mapping]
        keep = (
                ~local_free[:, None]
                & ~local_free[None, :]
            )

        row_parts.append(rows[keep])
        col_parts.append(cols[keep])
        value_parts.append(A_local[keep])

    row_parts.append(boundary_indices)
    col_parts.append(boundary_indices)

    value_parts.append(
        torch.ones(
            boundary_indices.numel(),
            dtype=value_parts[0].dtype,
            device=value_parts[0].device,
        )
    )
    sparse_indices = torch.stack([
        torch.cat(row_parts),
        torch.cat(col_parts),
    ])

    sparse_values = torch.cat(value_parts)

    A = torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=(nodal_points, nodal_points),
        dtype=sparse_values.dtype,
        device=sparse_values.device,
    ).coalesce()
    return A
        
'''
Take hat space points and compute gradient
'''
def gradhat_phihat(xhat: torch.Tensor, yhat: torch.Tensor, degree: int = 1):

    return torch.einsum(
            "ij,qjk->qik",
            poly_to_nodal_coords(degree),
            ref_pt_to_poly_gradient(xhat, yhat, degree)
            )
def phihat(xhat: torch.Tensor, yhat: torch.Tensor, degree: int = 1):
    return torch.einsum(
            "ij,qj->qi",
            poly_to_nodal_coords(degree),
            ref_pt_to_poly(xhat, yhat, degree)
            )

def edge_condition_to_dirichlet_mask(
        triangles: torch.Tensor,
        boundary_conditions: torch.Tensor,
        top_to_edge: dict,
        nodal_DoFs: int,
        edge_offset: int,
        degree: int = 1):
    dirichlet_mask = torch.zeros(nodal_DoFs, dtype=torch.bool)
    for idx, triangle in enumerate(triangles):
        edges = [triangle[1:], triangle[::2], triangle[:2]]
        boundary = boundary_conditions[idx]
        for inner_idx, edge in enumerate(edges):
            edge_tuple = (int(min(edge)), int(max(edge)))
            if boundary[inner_idx] == 1:
                dirichlet_mask[edge_tuple[0]] = True
                dirichlet_mask[edge_tuple[1]] = True
                edge_starting_loc = edge_offset + top_to_edge[edge_tuple] * (degree - 1)
                dirichlet_mask[edge_starting_loc:edge_starting_loc+(degree-1)] = True
    return dirichlet_mask

def local_to_global(triangle: torch.Tensor,
                    triangle_idx: int,
                    node_mapping: torch.Tensor,
                    top_to_edge: dict,
                    edge_offset: int,
                    internal_offset: int,
                    degree: int = 1):
    edges = [triangle[1:], triangle[::2], triangle[:2]]
    edges_tuple_list = []
    permutations = []
    for edge in edges:
        edges_tuple_list.append((min(edge).item(), max(edge).item()))
        if min(edge)!=edge[0]:
            permutations.append(True)
        else:
            permutations.append(False)


    num_pts_per_edge = degree-1
    num_internal_pts = (degree-1)*(degree-2) // 2
    mapping = torch.zeros_like(node_mapping)
    for idx, val in enumerate(node_mapping): 
        node_type = val % 10
        node_order = (val // 10) % 10
        if node_type == 1: # vertex
            global_vertex_number = triangle[node_order - 1]
            mapping[idx] = global_vertex_number 
        elif node_type == 2:
            if permutations[node_order-1]:
                mapping[idx] = (
                        (num_pts_per_edge-1) - 
                        (val//100 - 1) + 
                        top_to_edge[edges_tuple_list[node_order-1]]*num_pts_per_edge + 
                        edge_offset)
            else:
                mapping[idx] = (
                        (val//100 - 1) + 
                        top_to_edge[edges_tuple_list[node_order-1]]*num_pts_per_edge + 
                        edge_offset
                        )
        elif node_type == 3:
            mapping[idx] = (
                    internal_offset + 
                    triangle_idx*num_internal_pts + 
                    (node_order - 1)
                    )

    return mapping

def local_0_f(simplex: torch.Tensor,
              precomputed_phihats: torch.Tensor,
              quad_pts: torch.Tensor,
              weights: torch.Tensor):
    x1, y1 = simplex[0]
    x2, y2 = simplex[1]
    x3, y3 = simplex[2]

    # must be computed every local index, can replace with adjoint
    dx_dxhat = torch.Tensor([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
    dxhat_dx = torch.linalg.inv(dx_dxhat)
    scaling = torch.abs(torch.linalg.det(dx_dxhat))
    def f(x, y):
        return (
            2 * torch.pi**2
            * torch.sin(torch.pi * x)
            * torch.sin(torch.pi * y)
        )
    # integrate
    pts = torch.einsum(
            "ij,qj->qi",
            dx_dxhat,
            quad_pts) + simplex[0][None,...]

    f_evaled = f(pts[...,-2], pts[..., -1])

    quadrature_size, nodal_basis_size = precomputed_phihats.shape[:2]
    A = torch.zeros((nodal_basis_size), dtype=torch.float32)
    G = precomputed_phihats


    A = torch.einsum(
        "q,qi,q->i",
        weights,
        G,
        f_evaled,
    )

    return scaling * A

def global_0_f(topology: torch.Tensor,
               geometry: torch.Tensor,
               dirichlet_mask: torch.Tensor,
               node_mapping: torch.Tensor,
               top_to_edge: dict,
               edge_offset: int,
               internal_offset: int,
               precomputed_phihats: torch.Tensor,
               quad_pts: torch.Tensor,
               weights: torch.Tensor,
               degree: int = 1):
    num_internal_pts = (degree-1)*(degree-2) // 2
    free_nodal_points =  int((~dirichlet_mask).sum().item())
    nodal_points = internal_offset + num_internal_pts*topology.shape[0]
    row_parts = []
    value_parts = []

    for idx, points in enumerate(topology):
        simplex = torch.stack((geometry[points[0]], geometry[points[1]], geometry[points[2]]),
                              dim=0)
        A_local = local_0_f(simplex, precomputed_phihats, quad_pts, weights) 
        local_to_global_mapping = local_to_global(points,
                        idx,
                        node_mapping,
                        top_to_edge,
                        edge_offset,
                        internal_offset,
                        degree
                        )

        rows = local_to_global_mapping
        local_free = dirichlet_mask[local_to_global_mapping]
        keep = (
                ~local_free[:]
            )

        row_parts.append(rows[keep])
        value_parts.append(A_local[keep])

    sparse_indices = torch.stack([
        torch.cat(row_parts),
    ])

    sparse_values = torch.cat(value_parts)

    A = torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=(nodal_points,),
        dtype=sparse_values.dtype,
        device=sparse_values.device,
    ).coalesce()
    return A
     
