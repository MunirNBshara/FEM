import torch
from fem_helpers_2d import (
        pt_to_poly,
        poly_to_nodal_transform,
        ref_nodal_pts_cart,
        ref_to_simplex_transform,
        simplex_to_ref_transform)

'''
Generate interpolation of a fine grid using coarse basis
poly_to_nodal_transform after converting to polynomial on fine points would be a correct coarse_func

Inputs:
    fine_pts: torch.Tensor - location of fine grid points
    coarse_func: function evaluating the coarse grid basis
Returns:
    torch.Tensor - matrix obtaining interpolated nodal values on fine grid
'''
def one_level_local_prolongation(
        fine_geometry: torch.Tensor,
        coarse_geometry: torch.Tensor,
        # coarse_pts: torch.Tensor, 
        degree: int = 1 
        ) -> torch.Tensor:
    fine_in_coarse = fine_in_coarse_pts(
            fine_geometry,
            coarse_geometry,
            degree)
    fine_in_coarse_poly = pt_to_poly(
            fine_in_coarse[...,0],
            fine_in_coarse[...,1],
            degree)
    C = poly_to_nodal_transform(degree)
    P = torch.einsum("...ik,jk->...ij", fine_in_coarse_poly, C)
    return P

'''

'''
def fine_in_coarse_pts(
        fine_geometry: torch.Tensor,
        coarse_geometry: torch.Tensor,
        degree: int = 1
        ) -> torch.Tensor:

    # take reference points in x,y
    # convert to fine geometry using transform
    ref_pts = ref_nodal_pts_cart(degree)
    B_f, t_f = ref_to_simplex_transform(fine_geometry)
    B_c, t_c = simplex_to_ref_transform(coarse_geometry)
    fine_nodal_pts = (
        torch.einsum("...ij,...nj->...ni", B_f, ref_pts)
        + t_f.transpose(-2, -1)
    )

    fine_in_coarse_nodal_pts = (
        torch.einsum("...ij,...nj->...ni", B_c, fine_nodal_pts)
        + t_c.transpose(-2, -1)
    )

    return fine_in_coarse_nodal_pts


def full_prolongation(
        fine_geometry: torch.Tensor,
        coarse_geometry: torch.Tensor,
        fine_to_coarse_mapping: torch.Tensor,
        fine_global_mapping: torch.Tensor,
        coarse_global_mapping: torch.Tensor,
        num_fine_dofs: int,
        num_coarse_dofs: int,
        degree: int = 1) -> torch.Tensor:
    parent_geometry = coarse_geometry[fine_to_coarse_mapping]
    parent_dofs = coarse_global_mapping[fine_to_coarse_mapping]

    local_P = one_level_local_prolongation(
        fine_geometry,
        parent_geometry,
        degree
    )

    rows = fine_global_mapping[..., None].expand_as(local_P)
    cols = parent_dofs[:, None, :].expand_as(local_P)

    multiplicity = torch.bincount(
        fine_global_mapping.reshape(-1),
        minlength=num_fine_dofs,
    )

    row_weights = multiplicity[fine_global_mapping].to(local_P.dtype)
    values = local_P / row_weights[..., None]

    indices = torch.stack(
        (rows.reshape(-1), cols.reshape(-1)),
        dim=0,
    )

    return torch.sparse_coo_tensor(
        indices,
        values.reshape(-1),
        size=(num_fine_dofs, num_coarse_dofs),
    ).coalesce()
    # triangles by fine_nodes by coarse_nodes
    # based off triangle I can map the fine nodes(rows)
    # So need some mapping for coarse nodes(columns)
    # we can get that by embedding and mapping there.
    # row i should be indexed using fine_global_mapping
    # column j should be indexed using coarse_global_mapping
    # fine_global_mappign is num_triangles, num_local_dofs
    # same with coarse

'''
V-cycle general. Say you have mesh 0...n
repeat until k=0; start k=n, k=k-1
1. Solve to reduce high frequencies using smoother: A_k u_k = b_k
2. Form  b_k - A_ku_k = r_k.
3. Project both r_k and A_k into a coarser space. P^T A_k e_k = P^T r_k
4. Since this is underdetermined we lose information and thus write P^T A_k P P^T e_k approx P^T r_k
5. Let P^T r_k = b_(k-1) and P^T A_k P = A_(k-1) and P^T e_k = u_(k-1). u_(k-1) is a correction to u_k

at k=0 solve A_0 u_0 = b_0
repeat up until k
u_k = u_k + P u_(k-1)
optional post smoothing step meaning
compute r_k with this new u_k and solve for u_k again.
'''
from NLA_Algs.solve.classical_iterative_methods.jacobi import jacobi_step
class MG:
    V_l  = []
    A_l = []
    P_l = []
    levels = -1
    presmoothing_iters = 0
    postsmoothing_iters = 0

    def __init__(self, Vs, As: torch.Tensor, Ps, presmoothing, postsmoothing):
        self.levels = len(As)
        self.V_l = [Vs]
        self.A_l = [As.to_dense()]
        self.P_l = Ps 
        self.presmoothing_iters = presmoothing
        self.postmoothing_iters = postsmoothing

    def add_finer_subspace(self, one_level_embedding, V, A = None):
        self.levels += 1 
        V_c = self.V_l[-1] 
        self.P_l.append(
                full_prolongation(
                    V.geometry,
                    V_c.geometry,
                    one_level_embedding,
                    V.global_mapping,
                    V_c.global_mapping,
                    V.total_global_DoFs,
                    V_c.total_global_DoFs,
                    V.ref_element.degree).to_dense()
                )

        self.V_l.append(V)
        if A == None:
            A, F = self.V_l[-1].create_matrices()
        self.A_l.append(A.to_dense())

    def mu_cycle(self, r, l, mu):
        if l == 0:
            return torch.linalg.solve(self.A_l[0], r)
        # presmooth
        A = self.A_l[l]
        A_c = self.A_l[l - 1]
        P = self.P_l[l-1]
        e = torch.zeros_like(r)
        for _ in range(self.presmoothing_iters):
            e = jacobi_step(A, e, r) 

        r_c = P.T @ (r - A @ e)
        e_c = torch.zeros_like(r_c)
        for _ in range(mu):
            e_c += self.mu_cycle(r_c - A_c @ e_c, l-1, mu)
        # postsmooth
        e += P @ e_c
        for _ in range(self.postsmoothing_iters):
            e = jacobi_step(A, e, r)
        return e
        






    

