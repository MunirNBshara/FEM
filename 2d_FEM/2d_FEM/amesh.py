import torch
from meshview import MeshView

class AdaptiveMesh:
    geometry = torch.tensor([])
    cell_vertices = torch.tensor([])
    cell_parent = torch.tensor([]) 
    cell_children = torch.tensor([]) 
    cell_depth = torch.tensor([])
    cell_active = torch.tensor([])
    cell_refine_edge = torch.tensor([])
    cell_active_neighbors = torch.tensor([])
    cell_boundary_tags = torch.tensor([])
    edge_midpoint = {}
    vertex_count = 0
    cell_count = 0
    grid_level = 0

    def __init__(self, geometry, cell_vertices, boundary_tags):
        n_cells = cell_vertices.shape[0]

        self.geometry = geometry
        self.cell_vertices = cell_vertices.to(dtype=torch.long)

        self.cell_parent = torch.full((n_cells,), -1, dtype=torch.long)
        self.cell_children = torch.full((n_cells, 2), -1, dtype=torch.long)
        self.cell_depth = torch.zeros(n_cells, dtype=torch.long)
        self.cell_active = torch.ones(n_cells, dtype=torch.bool)

        self.cell_refine_edge = torch.zeros(n_cells, dtype=torch.long)
        self.edge_midpoint = {}

        self.cell_boundary_tags = boundary_tags
        self.vertex_count = n_cells + 2
        self.cell_count = n_cells

        self.revision = 0 

        self._views = {}

    def build_neighbors(self):
        active_cell_ids = torch.where(self.cell_active)[0]
        active_cell_vertices = self.cell_vertices[active_cell_ids]
        edge_pairs = torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=torch.long)
        edge_vertices = active_cell_vertices[:, edge_pairs]
        edge_reversed = edge_vertices[..., 0] > edge_vertices[..., 1]
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
        self.cell_active_neighbors = neighbors        

        return unique_edges, element_to_edge, counts, edge_reversed 

    def split_cell(self, cell_id):
        if torch.all(self.cell_children[cell_id] != -1):
            raise ValueError("You can only split leaf cells")

        local_cell_vertices = self.cell_vertices[cell_id]
        edge_num = self.cell_refine_edge[cell_id]
        
        edge_pairs = torch.tensor([[1, 2], [2, 0], [0, 1]], dtype=torch.long)
        edge_vertices = local_cell_vertices[edge_pairs[edge_num]].sort().values
        key = (edge_vertices[0].item(), edge_vertices[1].item())
        midpoint_id = self.vertex_count

        if key not in self.edge_midpoint:
            self.geometry = torch.cat((
                self.geometry,
                ( self.geometry[self.cell_vertices[cell_id, 1]] +
                 self.geometry[self.cell_vertices[cell_id, 2]] )[None, :]/2),
                                      0) 
            self.edge_midpoint[key] = midpoint_id
            self.vertex_count += 1
        else:
            midpoint_id = self.edge_midpoint[key]
        self.cell_vertices = torch.cat((
            self.cell_vertices,
            torch.tensor([[midpoint_id, 
                           self.cell_vertices[cell_id, 0], 
                           self.cell_vertices[cell_id, 1]
                           ],
                          [midpoint_id,
                           self.cell_vertices[cell_id, 2],
                           self.cell_vertices[cell_id, 0]
                           ]], dtype=self.cell_vertices.dtype)), 0)
        

        self.cell_children[cell_id, 0] = self.cell_count
        self.cell_children[cell_id, 1] = self.cell_count + 1
        
        self.cell_refine_edge = torch.cat((
            self.cell_refine_edge,
            torch.zeros(2, dtype=self.cell_refine_edge.dtype)), 0)
        self.cell_children = torch.cat((
            self.cell_children,
            -1 * torch.ones((2,2), dtype=self.cell_children.dtype)), 0)
        self.cell_count += 2
        self.cell_parent = torch.cat((
            self.cell_parent,
            torch.tensor([cell_id, cell_id],dtype=self.cell_parent.dtype)), 0)
        self.cell_depth = torch.cat((
            self.cell_depth, 
            torch.tensor([self.cell_depth[cell_id]+1, self.cell_depth[cell_id]+1],
                         dtype=self.cell_depth.dtype)), 0)
        self.cell_active = torch.cat((
            self.cell_active,
            torch.tensor([True, True],
                         dtype=self.cell_active.dtype)), 0)
        self.cell_active[cell_id] = False
        self.cell_boundary_tags = torch.cat((self.cell_boundary_tags,
            torch.Tensor([
                [self.cell_boundary_tags[cell_id, 2], self.cell_boundary_tags[cell_id, 0], 0],
                [self.cell_boundary_tags[cell_id, 1], 0, self.cell_boundary_tags[cell_id, 0]]
                ])), 0)

        return key

    def get_leafs(self, parent):
        children_id = self.cell_children[parent]
        if torch.all(children_id == -1):
            return [parent]
        children_list = []
        for i in children_id:
            children_list.extend(self.get_leafs(i))
        return children_list

    def refinement_step(self, tau_set):
        if torch.all(self.cell_children[tau_set] != -1):
            raise ValueError("You can only split leaf cells")
       
        bisected_edges_dict = {}
        for tau in tau_set:
            edge = self.split_cell(tau)
            if edge not in bisected_edges_dict:
                if self.cell_active_neighbors[tau, 0] == -1:
                    pass
                else:
                    bisected_edges_dict[edge] = self.cell_active_neighbors[tau, 0].item()
            else:
                bisected_edges_dict.pop(edge)

        # previous approach, single stepping some errors to debug
        while bisected_edges_dict.items():
            temp_rm = set() 
            temp_add = {} 
            for key in bisected_edges_dict:
                parent = bisected_edges_dict[key]
                leafs = torch.tensor(self.get_leafs(parent), dtype=torch.long)
                # find edges in leafs
                leaf_vertices = self.cell_vertices[leafs]
                condition = (torch.any(leaf_vertices==key[0], dim=-1) &
                             torch.any(leaf_vertices==key[1], dim=-1))
                if torch.any(condition):
                    edge = self.split_cell(leafs[condition].squeeze())
                    if edge not in bisected_edges_dict and edge not in temp_add:
                        if self.cell_active_neighbors[parent, 0] == -1:
                            pass
                        else:
                            temp_add[edge] = self.cell_active_neighbors[parent, 0]
                    else:
                        temp_rm.add(edge) 
    
            for add_edge in temp_add:
                bisected_edges_dict[add_edge] = temp_add[add_edge]
            for rm_edge in temp_rm:
                bisected_edges_dict.pop(rm_edge)
        self.revision += 1
    
    def build_view(self):
        if self.revision in self._views:
            raise ValueError("This revision already has a snapshot")
        unique_edges, element_to_edge, counts, edge_reversed = self.build_neighbors()
        cell_ids = torch.where(self.cell_active)[0] 
        snapshot = MeshView(self,
                            self.revision,
                            cell_ids)
        snapshot.build_relations(element_to_edge,
                                 edge_reversed,
                                 self.vertex_count,
                                 unique_edges.shape[0])

        self._views[self.revision] = snapshot
        return snapshot

    def view(self, revision=None):
        if revision is None:
            revision = self.revision
        return self._views[revision]
     
    def one_level_embedding(self, coarse_layer, fine_layer):
        coarse_view = self.view(coarse_layer)
        fine_view = self.view(fine_layer)

        
        fine_ids = fine_view.cell_ids
        curr_ids = fine_ids.clone()
        
        coarse_ids = coarse_view.cell_ids
        coarse_mask = -1 * torch.ones(max(fine_ids) + 1, dtype=torch.long)
        indices = torch.arange(coarse_ids.shape[0], dtype=torch.long)
        coarse_mask[coarse_ids] = indices

        while True:
            finished = (coarse_mask[curr_ids] != -1)

            if finished.all():
                break

            curr_ids = torch.where(
                finished,
                curr_ids,
                self.cell_parent[curr_ids],
            )


        return curr_ids, coarse_mask[curr_ids]

   

    
        
        






