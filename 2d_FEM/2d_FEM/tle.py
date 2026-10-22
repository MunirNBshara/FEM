import torch
from fem_helpers_2d import (
        ref_nodal_pts_cart,
        poly_to_nodal_transform,
        pt_to_poly,
        pt_to_poly_grad,
        pt_to_poly_hessian,
        ref_nodal_pt_to_identity_detailed,
        pts_bary_to_pts,
        )
from fem_quadrature_2d import (
        dunavant_rules,
        gaussian_quadrature
        )

class TriangleLagrangeElement:
    
    def __init__(self, degree):
        self.degree = degree
        self.local_dof_identifiers = (
            ref_nodal_pt_to_identity_detailed(degree)
        )
        self.ref_nodal_pts_cart = ref_nodal_pts_cart(degree)
        self.poly_to_nodal_transform = poly_to_nodal_transform(degree)
        triangle_quad_pts_bary, self.triangle_quad_weights = dunavant_rules(degree)
        self.triangle_quad_pts = pts_bary_to_pts(triangle_quad_pts_bary)
        self.edge_quad_pts, self.edge_quad_weights = gaussian_quadrature(degree)
        self.node_mapping = ref_nodal_pt_to_identity_detailed(self.degree) 

    def ref_pts_cart_to_nodal_basis(self, pts): 
        return torch.einsum(
                "ij,...j->...i",
                self.poly_to_nodal_transform,
                pt_to_poly(pts[..., 0], pts[..., 1], self.degree)
                )

    def grad_ref_pts_cart_to_nodal_basis(self, pts):
        return torch.einsum(
                "ij,...jk->...ik",
                self.poly_to_nodal_transform,
                pt_to_poly_grad(pts[..., 0], pts[..., 1], self.degree)
                )

    def hessian_ref_pts_cart_to_nodal_basis(self, pts):
        return torch.einsum(
                "ij,...jkl->...ikl",
                poly_to_nodal_transform(self.degree),
                pt_to_poly_hessian(pts[..., 0], pts[..., 1], self.degree)
                )

    def gradhat_edge(self, xhat: torch.Tensor) -> torch.Tensor:    
        edge1_quad_pts = torch.stack((xhat,1-xhat), dim=-1)
        edge2_quad_pts = torch.stack((torch.zeros(xhat.shape[0]), xhat), dim=-1)
        edge3_quad_pts = torch.stack((xhat,torch.zeros(xhat.shape[0])), dim=-1)

        edge1_grad = self.grad_ref_pts_cart_to_nodal_basis(edge1_quad_pts)
        edge2_grad = self.grad_ref_pts_cart_to_nodal_basis(edge2_quad_pts)
        edge3_grad = self.grad_ref_pts_cart_to_nodal_basis(edge3_quad_pts)
       
        gradhat = torch.stack((edge1_grad, edge2_grad, edge3_grad), dim = -1)
        return gradhat

    @property
    def nodal_at_triangle_quad_pts(self):
        return self.ref_pts_cart_to_nodal_basis(self.triangle_quad_pts)

    @property
    def grad_nodal_at_triangle_quad_pts(self):
        return self.grad_ref_pts_cart_to_nodal_basis(self.triangle_quad_pts)
    
    @property
    def hessian_nodal_at_triangle_quad_pts(self):
        return self.hessian_ref_pts_cart_to_nodal_basis(self.triangle_quad_pts)

    @property
    def grad_nodal_at_triangle_edge_quad_pts(self):
        return self.gradhat_edge(self.edge_quad_pts)

    

    


        


    
