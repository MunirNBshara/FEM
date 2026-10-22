import torch
from vectorized_fem_helpers_2d import (
        vectorized_edge_condition_to_dirichlet_mask, 
        vectorized_local_to_global,
        vectorized_global_1_1_stripped,
        vectorized_global_0_f_stripped,
        )

def forcing(x, y):
    return (
        2 * torch.pi**2
        * torch.sin(torch.pi * x)
        * torch.sin(torch.pi * y)
    )

class FESpace:
   
    def __init__(self,
                 meshview,
                 ref_element):
        self.meshview = meshview
        self.ref_element = ref_element

        self.non_internal_DoFs = (self.meshview.vertex_count +
                                  self.meshview.edge_count * 
                                  (self.ref_element.degree - 1))

        self.total_global_DoFs = (self.non_internal_DoFs + 
                                  ((self.ref_element.degree - 1) * 
                                   (self.ref_element.degree - 2) // 2) *
                                  self.meshview.topology.shape[0])

        self.global_mapping = vectorized_local_to_global(
                self.meshview.topology,
                self.meshview.element_to_edge,
                self.meshview.edge_reversed,
                self.ref_element.node_mapping,
                self.meshview.vertex_count,
                self.non_internal_DoFs,
                self.ref_element.degree)

    @property
    def forcing(self):
        return forcing 
    @property
    def topology(self):
        return self.meshview.topology

    @property
    def geometry(self):
        return self.meshview.geometry

    @property
    def boundary_tags(self):
        return self.meshview.boundary_tags

    def create_matrices(self):
        topology = self.meshview.topology
        geometry = self.meshview.geometry
        boundary_tags = self.meshview.boundary_tags
        vertex_count = self.meshview.vertex_count
        
        dirichlet_mask = vectorized_edge_condition_to_dirichlet_mask(
                topology,
                boundary_tags,
                self.meshview.element_to_edge,
                vertex_count,
                self.total_global_DoFs,
                self.ref_element.degree)

        F_sparse = vectorized_global_0_f_stripped(
                geometry,
                dirichlet_mask,
                self.global_mapping,
                self.ref_element.nodal_at_triangle_quad_pts,
                self.ref_element.triangle_quad_pts,
                self.ref_element.triangle_quad_weights,
                forcing
                )
        A_sparse = vectorized_global_1_1_stripped(
                geometry,
                dirichlet_mask,
                self.global_mapping,
                self.ref_element.grad_nodal_at_triangle_quad_pts,
                self.ref_element.triangle_quad_weights
                )

        return A_sparse, F_sparse
        


    
