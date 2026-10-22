import torch

class MeshView:
    def __init__(
        self,
        mesh,
        revision,
        cell_ids,
        ):
        self.mesh = mesh
        self.revision = revision
        self.cell_ids = cell_ids

    def build_relations(
        self,
        element_to_edge,
        edge_reversed,
        vertex_count,
        edge_count
        ):
        self.element_to_edge = element_to_edge
        self.edge_reversed = edge_reversed
        self.vertex_count = vertex_count
        self.edge_count = edge_count

    @property
    def topology(self):
        return self.mesh.cell_vertices[self.cell_ids]

    @property
    def geometry(self):
        return self.mesh.geometry[self.topology]

    @property
    def boundary_tags(self):
        return self.mesh.cell_boundary_tags[self.cell_ids]

