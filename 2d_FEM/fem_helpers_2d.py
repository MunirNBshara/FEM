import torch
import math


'''
Generates polynomial from reference point.
Ordering: lowest->highest degree, powers shift from x->y.

Inputs:
    x: torch.Tensor - x coord
    y: torch.Tensor - y coord
    degree: int = 1 - polynomial degree
Output:
    torch.Tensor - evaluated polynomial
'''
def pt_to_poly(x: torch.Tensor, y: torch.Tensor, degree: int = 1) -> torch.Tensor:
    poly = []
    for i in range(degree+1):
        for j in range(i+1):
            poly.append(x**(i-j) * y**(j))
    return torch.stack(poly, dim=-1)

'''
Evaluates gradient of polynomial.
Ordering: lowest->highest degree, powers shift from x->y

Inputs:
    x: torch.Tensor - x coord
    y: torch.Tensor - y coord
    degree: int = 1 - polynomial degree
Outputs:
    torch.Tensor - Nd by 2 matrix, first column dx, second dy
'''
def pt_to_poly_grad(x: torch.Tensor, y:torch.Tensor, degree: int = 1) -> torch.Tensor:
    terms = (degree + 1) * (degree + 2) // 2
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
def pt_to_poly_hessian(x: torch.Tensor, y:torch.Tensor, degree: int = 1) -> torch.Tensor:
    terms = (degree + 1) * (degree + 2) // 2
    poly_grad_x = torch.zeros(x.shape[-1], terms, 2, dtype = x.dtype)
    poly_grad_y = torch.zeros(y.shape[-1], terms, 2, dtype = y.dtype)
    idx = 0
    for i in range(degree+1):
        for j in range(i+1):
            if j!=0 and (i-j)!=0:
                poly_grad_y[..., idx, 0] = ((i-j)*x**(i-j-1) * j * y**(j-1))
                poly_grad_x[..., idx, 1] = ((i-j)*x**(i-j-1) * j * y**(j-1))
            if j>1:
                poly_grad_y[..., idx, 1] = (x**(i-j) * j * (j-1) * y**(j-2))
            if (i-j)>0:
                poly_grad_x[..., idx, 0] = ((i-j)*(i-j-1)*x**(i-j-2) * y**(j))
            idx+=1
    return torch.stack((poly_grad_x,poly_grad_y), dim=-1)


'''
Generates nodal points on reference simplex for an nth degree trial basis
Generation is based on barycentric coordinates where
    i/deg + j/deg + k/deg = 1
Ordering: increasing L1 norm with equivalent norm being ordered by ascending y

Inputs:
    degree: int = 1 - polynomial degree
Outputs:
    torch.Tensor - rational points in [0,1] x [0,1-x](triangular domain) 
'''
def ref_nodal_pts_cart(degree: int = 1) -> torch.Tensor:
    nodes = []

    for i in range(degree, -1, -1):
        for j in range(degree-i, -1, -1):
            k = degree - i - j
            nodes.append((j / degree, k / degree))

    return torch.tensor(nodes)

'''
Same as ref_nodal_pts_cart but in barycentric coordinates.

Inputs:
    degree: int = 1 - polynomial degree
Outputs:
    torch.Tensor - rational points in [0,1]^3
'''
def ref_nodal_pts_bary(degree: int = 1) -> torch.Tensor:
    nodes = []

    for i in range(degree, -1, -1):
        for j in range(degree-i,-1,-1):
            k = degree - i - j
            nodes.append((1 - i / degree, j / degree, k / degree))

    return torch.tensor(nodes)
'''
Identifies where the nodal point lay on a simplex.
Mapping(where the point lies):
    1 - Vertex
    2 - Edge
    3 - Internal

Inputs:
    pts_bary: torch.Tensor - points in barycentric coordinates
    tol: float(1e-6) - zero tolerance
Output:
    torch.Tensor - pts_bary.shape[-2] with identities listed out 
'''
def pts_bary_to_identity(pts_bary: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    identities = torch.zeros(pts_bary.shape[-2])
    for idx in range(pts_bary.shape[-2]):
        pt = pts_bary[..., idx, :]
        nz_mask = torch.where(abs(pt)>tol, 1.0,0.0)
        nnz = torch.sum(nz_mask, dim =-1)
        if nnz < 1:
            identities[idx] = 3
        elif nnz < 2:
            identities[idx] = 2
        else:
            identities[idx] = 1
    return identities

'''
Identify detailed location of points as dictated by ref_nodal_pts_cart.
Mapping(where the point lies):
    Units:
        1 - Vertex
        2 - Edge 
        3 - Internal
    Tens: 
        Vertex numbering - 1,2,3 IN SAME ORDER AS NODAL POINT CONSTRUCTION
        Edge numbering - side opposite to vertex of the same number
        Internal numbering - SAME ORDER AS NODAL POINT CONSTRUCTION
    Hundreds:
        Vertex/Internal - No meaning
        Edge numbering - order within the edge. SAME ORDER AS NODAL POINT CONSTRUCTION

Inputs:
    degree: int(1) - degree of nodal basis
Outputs:
    torch.Tensor - detailed point reference
'''
def ref_nodal_pt_to_identity_detailed(degree: int = 1) -> torch.Tensor:
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
Generates appropriate matrix transform from cartesian polynomial to nodal coord

Takes reference nodal pts -> converts to polynomial -> 
    solves for matrix which when applied to polynomial gives delta_mn

Inputs:
    degree: int(1) - nodal degree
Outputs:
    torch.Tensor - square matrix which converts cartesian polynomial to nodal coords
'''
def poly_to_nodal_transform(degree: int = 1) -> torch.Tensor:
    nodes = ref_nodal_pts_cart(degree)
    xhat = nodes[:, 0]
    yhat = nodes[:, 1]

    V = pt_to_poly(xhat, yhat, degree).T
    C = torch.linalg.solve(V, torch.eye(
        V.shape[0],
    ))

    return torch.where(C.abs() > 1e-6, C, torch.zeros_like(C))

'''
Index in cartesian polynomial based x and y degrees.
Ordering: lowest->highest degree, powers shift from x->y

Inputs:
    x_power: int
    y_power: int
Returns:
    int - Index relative to 0.
'''
def cartesian_index(x_power: int, y_power: int) -> int:
    total = x_power + y_power
    return total * (total + 1) // 2 + y_power

'''
Index in barycentric polynomial based on lambda_i degrees
Ordering: i order followed by j followed by k.

Inputs:
    i: int - vertex 1 power
    j: int - vertex 2 power
    k: int - vertex 3 power
    degree: int - nodal degree
Outputs:
    int - Index relative to 0.
'''
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
Generates transform to go from cartesian polynomial to barycentric polynomial

Inputs:
    degree: int(1) - nodal degree
Outputs:
    torch.Tensor - change of basis matrix 
'''
def poly_to_bpoly_transform(degree: int = 1) -> torch.Tensor:
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
Converts barycentric polynomial to barycentric points
This could be made more accurate using information from cross terms
Inputs:
    bary: torch.Tensor - barycentric polynomial to convert
    degree: int(1) - degree of polynomial 
Outputs:
    torch.Tensor - [lambda_1, lambda_2, lambda_3]
'''
def bpoly_to_pts_bary(bpoly: torch.Tensor, degree: int = 1) -> torch.Tensor:
    lambda1_d = bpoly[..., 0]
    lambda2_d = bpoly[..., degree * (degree + 1) // 2]
    lambda3_d = bpoly[..., -1]

    pure_powers = torch.stack(
        [lambda1_d, lambda2_d, lambda3_d],
        dim=-1,
    )

    return pure_powers.clamp_min(0).pow(1.0 / degree)

'''
Converts barycentric points to barycentric polynomial

Inputs:
    pts_bary: torch.Tensor - barycentric points to be converted
    degree: int(1) - degree of polynomial
Outputs:
    torch.Tensor - barycentric polynomial
'''
def pts_bary_to_bpoly(pts_bary: torch.Tensor, degree: int = 1):
    lambda1 = pts_bary[...,0]
    lambda2 = pts_bary[...,1]
    lambda3 = pts_bary[...,2]
    poly = []
    for i in range(degree, -1, -1):
        for j in range(degree-i, -1, -1):
            k = degree - j - i 
            poly.append(lambda1**i * lambda2**j * lambda3**k)
    return torch.stack(poly, dim=-1)

'''
Converts barycentric polynomial to nodal points

Inputs:
    bpoly: torch.Tensor - barycentric polynomial
    degree: int(1) - nodal and barycentric degree
Outputs:
    torch.Tensor - nodal points 
'''
def bpoly_to_nodal(bpoly: torch.Tensor, degree: int) -> torch.Tensor:
    A = poly_to_bpoly_transform(degree)
    C = poly_to_nodal_transform(degree)

    poly_cartesian = torch.linalg.solve(A, bpoly)
    return C @ poly_cartesian

'''
Converts barycentric pts to cartesian pts

Inputs: 
    pts_bary: torch.Tensor - barycentric polynomial
Outputs:
    torch.Tensor - cartesian coordinates
'''
def pts_bary_to_pts(pts_bary: torch.Tensor) -> torch.Tensor:
    return torch.stack((pts_bary[...,-2],pts_bary[...,-1]), dim=-1)

'''
Takes simplex, outputs local stiffness matrix

Inputs: 
    simplex: torch.Tensor - simplex to integrate over
    precomputed_gradhat_phihats: torch.Tensor - precomputed transforms
Outputs:
    torch.Tensor - integral over one simplex
'''
def local_1_1(simplex: torch.Tensor,
              precomputed_gradhat_phihats: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    x1, y1 = simplex[0]
    x2, y2 = simplex[1]
    x3, y3 = simplex[2]

    # must be computed every local index, can replace with adjoint
    # dx_dxhat = torch.Tensor([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
    # dxhat_dx = torch.linalg.inv(dx_dxhat)
    # scaling = torch.abs(torch.linalg.det(dx_dxhat))
    adjugate = torch.Tensor([[y3 - y1, -(x3 - x1)], [-(y2 - y1), x2 - x1]])
    scaling = 1 / torch.abs(adjugate[0,0]*adjugate[1,1] - adjugate[0,1]*adjugate[1,0])

    # integrate
    G = precomputed_gradhat_phihats @ adjugate

    A = torch.einsum(
        "q,qia,qja->ij",
        weights,
        G,
        G,
    )

    return scaling * A
'''
Obtain global stiffness matrix

Inputs:
    topology: torch.Tensor - connectivity for each triangle
    geometry: torch.Tensor - location of each vertex 
    dirichlet_mask: torch.Tensor - nodal points to be excluded from the construction
    node_mapping: torch.Tensor - mapping of nodal points to type of point 
    top_to_edge: dict - edge vertices to edge number 
    edge_offset: int - number of vertices
    internal_offset: int - number of vertices + number of edges
    precomputed_gradhat_phihats: torch.Tensor - precomputed quadrature simplex irrespective transform
    weights: torch.Tensor - weights for quadrature weights
    degree: int(1) - nodal degree
Outputs:
    torch.Tensor - Sparse COO matrix
'''
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
               ) -> torch.Tensor:
    num_internal_pts = (degree-1)*(degree-2) // 2
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
Construct derivative of phihat w.r.t. the reference coordinates

Inputs:
    xhat: torch.Tensor - xhat point
    yhat: torch.Tensor - yhat point
    degree: int(1) - nodal degree 
Outputs:
    torch.Tensor - derivative matrix
'''
def gradhat_phihat(xhat: torch.Tensor, yhat: torch.Tensor, degree: int = 1) -> torch.Tensor:

    return torch.einsum(
            "ij,qjk->qik",
            poly_to_nodal_transform(degree),
            pt_to_poly_grad(xhat, yhat, degree)
            )

'''
Compute phihat w.r.t reference coordinates
Inputs:
    xhat: torch.Tensor - xhat point
    yhat: torch.Tensor - yhat point
    degree: int(1) - nodal degree 
Outputs:
    torch.Tensor - derivative matrix
'''
def phihat(xhat: torch.Tensor, yhat: torch.Tensor, degree: int = 1) -> torch.Tensor:
    return torch.einsum(
            "ij,qj->qi",
            poly_to_nodal_transform(degree),
            pt_to_poly(xhat, yhat, degree)
            )

def hessianhat(xhat: torch.Tensor, yhat: torch.Tensor, degree: int = 1) -> torch.Tensor:
    return torch.einsum(
            "ij,qjkl->qikl",
            poly_to_nodal_transform(degree),
            pt_to_poly_hessian(xhat, yhat, degree)
            )

    

    
def gradhat_edge(xhat: torch.Tensor, degree: int = 1) -> torch.Tensor:
    
    edge1_quad_pts = torch.stack((xhat,1-xhat), dim=-1)
    edge2_quad_pts = torch.stack((torch.zeros(xhat.shape[0]), xhat), dim=-1)
    edge3_quad_pts = torch.stack((xhat,torch.zeros(xhat.shape[0])), dim=-1)

    edge1_grad = gradhat_phihat(edge1_quad_pts[...,-2], edge1_quad_pts[...,-1], degree)
    edge2_grad = gradhat_phihat(edge2_quad_pts[...,-2], edge2_quad_pts[...,-1], degree)
    edge3_grad = gradhat_phihat(edge3_quad_pts[...,-2], edge3_quad_pts[...,-1], degree)
   
    gradhat = torch.stack((edge1_grad, edge2_grad, edge3_grad), dim = -1)
    return gradhat

'''
Converts edge condition matrix list to dirichlet mask

Inputs:
    triangles: torch.Tensor - topology of triangles
    boundary_conditions: torch.Tensor - Type of boundary condition list
    top_to_edge: dict - given edge give me global index
    nodal_DoFs: int - number of global degrees of freedom
    edge_offset: int - total number of vertices
    degree: int(1) - degree of nodal points
Outputs:
    torch.Tensor - boolean tensor where dirichlet conditions take place
'''
def edge_condition_to_dirichlet_mask(
        triangles: torch.Tensor,
        boundary_conditions: torch.Tensor,
        top_to_edge: dict,
        nodal_DoFs: int,
        edge_offset: int,
        degree: int = 1) -> torch.Tensor:
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

'''
Converts local index to global index.

Inputs:
    triangle: torch.Tensor - a single triangle
    triangle_idx: int - which triangle
    node_mapping: torch.Tensor - type of node
    top_to_edge: dict - given edge give me global index
    edge_offset: int - total number of vertices
    internal_offset: int - total number of vertices + total number of edges
    degree: int(1) - degree of nodal points
Outputs:
    torch.Tensor - mapping array
'''
def local_to_global(triangle: torch.Tensor,
                    triangle_idx: int,
                    node_mapping: torch.Tensor,
                    top_to_edge: dict,
                    edge_offset: int,
                    internal_offset: int,
                    degree: int = 1) -> torch.Tensor:
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

def f(x, y):
    return (
        2 * torch.pi**2
        * torch.sin(torch.pi * x)
        * torch.sin(torch.pi * y)
    )
    
'''
Local forcing matrix

Inputs:
    simplex: torch.Tensor - geometric position of triangle
    precomputed_phihats: torch.Tensor - precomputed phihat functions
    quad_pts: torch.Tensor - quadrature pts 
    weights: torch.Tensor - quadrature weights 
    
Outputs:
    torch.Tensor - local forcing function
'''
def local_0_f(simplex: torch.Tensor,
              precomputed_phihats: torch.Tensor,
              quad_pts: torch.Tensor,
              weights: torch.Tensor,
              forcing) -> torch.Tensor:
    x1, y1 = simplex[0]
    x2, y2 = simplex[1]
    x3, y3 = simplex[2]

    # must be computed every local index, can replace with adjoint
    # dx_dxhat = torch.Tensor([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
    # dxhat_dx = torch.linalg.inv(dx_dxhat)
    # scaling = torch.abs(torch.linalg.det(dx_dxhat))
    adjugate = torch.Tensor([[y3 - y1, -(x3 - x1)], [-(y2 - y1), x2 - x1]])
    scaling = torch.abs(adjugate[0,0]*adjugate[1,1] - adjugate[1,0]*adjugate[0,1])

    # integrate
    pts = torch.einsum(
            "ij,qj->qi",
            adjugate,
            quad_pts) + simplex[0][None,...]

    f_evaled = forcing(pts[...,-2], pts[..., -1])

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
'''
Obtain global forcing matrix

Inputs:
    topology: torch.Tensor - connectivity for each triangle
    geometry: torch.Tensor - location of each vertex 
    dirichlet_mask: torch.Tensor - nodal points to be excluded from the construction
    node_mapping: torch.Tensor - mapping of nodal points to type of point 
    top_to_edge: dict - edge vertices to edge number 
    edge_offset: int - number of vertices
    internal_offset: int - number of vertices + number of edges
    precomputed_gradhat_phihats: torch.Tensor - precomputed quadrature simplex irrespective transform
    quad_pts: torch.Tensor, points for quadrature
    weights: torch.Tensor - weights for quadrature weights
    degree: int(1) - nodal degree
Outputs:
    torch.Tensor - Sparse COO matrix
'''
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
               forcing,
               degree: int = 1):
    num_internal_pts = (degree-1)*(degree-2) // 2
    free_nodal_points =  int((~dirichlet_mask).sum().item())
    nodal_points = internal_offset + num_internal_pts*topology.shape[0]
    row_parts = []
    value_parts = []

    for idx, points in enumerate(topology):
        simplex = torch.stack((geometry[points[0]], geometry[points[1]], geometry[points[2]]),
                              dim=0)
        A_local = local_0_f(simplex, precomputed_phihats, quad_pts, weights, forcing) 
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


