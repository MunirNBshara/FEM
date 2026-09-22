
import torch


def mark_refinement(etas: torch.Tensor, theta: float = 0.5):
    etas_squared = etas.square()
    eta_domain = torch.sqrt(etas_squared.sum())
    sorted_eta_squared, triangle_order = torch.sort(etas_squared, descending=True)
    cumulative_error = torch.cumsum(sorted_eta_squared, dim=0)

    infinum = theta * eta_domain
    number_marked = torch.searchsorted(cumulative_error, infinum).item() + 1 
    marked_triangles = triangle_order[:number_marked]

    return marked_triangles


'''
Uses Dörfler metric to identify where refinement must be done:
    eta^2(u_T, M_T) >= theta eta^2(u_T, T)
where eta(u_T, M_T) = sqrt( sum_tau eta^2(u_T, tau))
and eta(u_T, tau) = sqrt( norm_(0, tau)[h (f + laplacian u_T)]^2 + sum_edges norm_(0, edge)[h^0.5 *(grad u_T dot normal_edge)]^2)
'''
def compute_eta(topology: torch.Tensor,
                geometry: torch.Tensor,
                quad_pts_tauhat: torch.Tensor,
                quad_weights_tauhat: torch.Tensor,
                quad_weights_edge: torch.Tensor,
                global_mapping: torch.Tensor,
                u: torch.Tensor,
                element_to_edge: torch.Tensor,
                forcing,
                boundary_conditions: torch.Tensor,
                precomputed_hessianhat: torch.Tensor,
                precomputed_gradhat: torch.Tensor,
                edge_elements: int):
    vertices = geometry[topology]

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
        precomputed_gradhat,
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


class AdaptiveMesh:
    geometry = torch.tensor([]) 
    cell_vertices = torch.tensor([]) 
    cell_parent = torch.tensor([]) 
    cell_children = torch.tensor([]) 
    cell_level = torch.tensor([])
    cell_active = torch.tensor([])
    cell_refine_edge = torch.tensor([])
    cell_neighbors = torch.tensor([])
    cell_dirichlet = torch.tensor([])
    edge_midpoint = {}
    vertex_count = 0
    cell_count = 0

    def __init__(self, geometry_top, topology_top, dirichlet_top):
        self.geometry = geometry_top
        self.vertex_count = geometry_top.shape[0]
        self.cell_vertices = topology_top.to(torch.long)
        self.cell_count = topology_top.shape[0]
        self.cell_parent = -1 * torch.ones(topology_top.shape[0], dtype=torch.long)
        self.cell_children = -1 * torch.ones(topology_top.shape[0],2, dtype=torch.long)
        self.cell_level = torch.zeros(topology_top.shape[0])
        self.cell_active = torch.ones(topology_top.shape[0], dtype=torch.bool)
        self.cell_refine_edge = torch.zeros(topology_top.shape[0], dtype=torch.long)
        self.cell_dirichlet = dirichlet_top 
        self.build_neighbors()

        # edge_midpoint gets added to if there is a split

    def build_neighbors(self):
        active_cell_ids = torch.where(self.cell_active)[0]
        active_cell_vertices = self.cell_vertices[active_cell_ids]
        edge_pairs = torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=torch.long)
        edge_vertices = active_cell_vertices[:, edge_pairs]
        canonical_edges = edge_vertices.sort(dim=-1).values

        flat_edges = canonical_edges.reshape(-1, 2)

        unique_edges, inverse, counts = torch.unique(
            flat_edges,
            dim=0,
            return_inverse=True,
            return_counts=True
        )

        element_to_edge = inverse.reshape(active_cell_vertices.shape[0], 3)
        order = torch.argsort(inverse)
        group_starts = torch.cumsum(counts, dim=0) - counts

        interior_groups = torch.where(counts == 2)[0] 
        first = order[group_starts[interior_groups]]
        second = order[group_starts[interior_groups] + 1]

        first_active_cell = first // 3
        first_local_edge = first % 3
        second_active_cell = second // 3
        second_local_edge = second % 3

        neighbors = torch.full((self.cell_vertices.shape[0], 3), -1)

        first_cell_id = active_cell_ids[first_active_cell]
        second_cell_id = active_cell_ids[second_active_cell]
        neighbors[first_cell_id, first_local_edge] = second_cell_id
        neighbors[second_cell_id, second_local_edge] = first_cell_id

        self.cell_neighbors = neighbors

        return unique_edges, element_to_edge, counts


    # Single cell updates
    def split_cell(self, cell_id):
        if torch.all(self.cell_children[cell_id] != -1):
            raise ValueError("You can only split leaf cells")
        local_cell_vertices = self.cell_vertices[cell_id]
        edge_num = self.cell_refine_edge[cell_id]
        
        edge_pairs = torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=torch.long)
        edge_vertices = local_cell_vertices[edge_pairs[edge_num]].sort().values
        key = (edge_vertices[0].item(),edge_vertices[1].item())
        midpoint_id = self.vertex_count
        if key not in self.edge_midpoint:
            self.geometry = torch.cat(
                (self.geometry,
                 (self.geometry[self.cell_vertices[cell_id,1]] + self.geometry[self.cell_vertices[cell_id,2]])[None,:]/2),
                0)    
            self.edge_midpoint[key] = midpoint_id
            self.vertex_count+=1
        else:
            midpoint_id = self.edge_midpoint[key]
        self.cell_vertices = torch.cat((
            self.cell_vertices,
            torch.tensor([[midpoint_id, self.cell_vertices[cell_id,0], self.cell_vertices[cell_id,1]],
                         [midpoint_id, self.cell_vertices[cell_id,2], self.cell_vertices[cell_id,0]]], dtype=self.cell_vertices.dtype)),
            0)
        

        self.cell_children[cell_id,0] = self.cell_count
        self.cell_children[cell_id,1] = self.cell_count+1
        
        self.cell_refine_edge = torch.cat((self.cell_refine_edge, torch.zeros(2, dtype=self.cell_refine_edge.dtype)), 0)
        self.cell_children = torch.cat((
            self.cell_children,
            -1 * torch.ones((2,2), dtype=self.cell_children.dtype)), 0)
        self.cell_count+=2
        self.cell_parent = torch.cat((self.cell_parent, torch.tensor([cell_id,cell_id],dtype=self.cell_parent.dtype)), 0)
        self.cell_level = torch.cat(
                (self.cell_level, 
                 torch.tensor([self.cell_level[cell_id]+1,self.cell_level[cell_id]+1], dtype=self.cell_level.dtype)),
                0)
        self.cell_active = torch.cat((self.cell_active, torch.tensor([True,True], dtype=self.cell_active.dtype)), 0)
        self.cell_active[cell_id] = False
        self.cell_dirichlet = torch.cat((self.cell_dirichlet,
            torch.Tensor([[self.cell_dirichlet[cell_id,2], self.cell_dirichlet[cell_id,0], 0],
                          [self.cell_dirichlet[cell_id,1], 0, self.cell_dirichlet[cell_id,0]]])),0
            )
        return key

    
    def get_leafs(self, parent):
        children_id = self.cell_children[parent]
        if torch.all(children_id == -1):
            return [parent]
        children_list = []
        for i in children_id:
            children_list.extend(self.get_leafs(i))
        return children_list
    # Take a set of taus
    # refine them
    # Look at what neighbors/children require refining
    # We can do that by looking at neighbor and self and ask whether edge_midpoint exists

    def refinement_step(self, tau_set):
        if torch.all(self.cell_children[tau_set] != -1):
            raise ValueError("You can only split leaf cells")
       
        bisected_edges_dict = {}
        for tau in tau_set:
            edge = self.split_cell(tau)
            if edge not in bisected_edges_dict:
                if self.cell_neighbors[tau,0] == -1:
                    pass
                else:
                    bisected_edges_dict[edge] = self.cell_neighbors[tau,0].item()
            else:
                bisected_edges_dict.pop(edge)

        while bisected_edges_dict.items():
            temp_rm = []
            temp_add = []
            for key in bisected_edges_dict:
                parent = bisected_edges_dict[key]
                leafs = torch.tensor(self.get_leafs(parent), dtype=torch.long)
                # find edges in leafs
                leaf_vertices = self.cell_vertices[leafs]
                condition = torch.any(leaf_vertices==key[0], dim=-1) & torch.any(leaf_vertices==key[1], dim=-1) 
                edge = self.split_cell(leafs[condition].squeeze())
                if edge not in bisected_edges_dict:
                    if self.cell_neighbors[parent, 0] == -1:
                        pass
                    else:
                        temp_add.append((edge, self.cell_neighbors[parent,0]))
                else:
                    temp_rm.append(edge)
            for rm_edge in temp_rm:
                bisected_edges_dict.pop(rm_edge)
            for add_edge in temp_add:
                bisected_edges_dict[add_edge[0]] = add_edge[1]


    

    
                









                    
        # while cond
        # split the set
        # this will create a new set of edges which are bisected.
        # if the bisected edges match those in the active set then remove it
        # if the bisected edges dont match append as well as parent triangle idx they affect
        # for each of the left over bisected edge, go down the leaves until a triangle with that edge is found
        # repeat
    
    
    

        

        
        
        


        


        

    







    
    


    


    
    


    

    
    

