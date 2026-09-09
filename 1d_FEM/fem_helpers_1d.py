import torch

'''
In 1d, the reference simplex namely the unit segment can only 2 degrees of freedom.
The first is scaling meaning length of the segment.
The second is translation along the 1d space.

This function inverses the 1d affin transformation to find jacobian
    ax+b
'''
def find_jacobian(topology: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    # Signed Jacobian of F(xi) = J*xi + b
    return geometry[topology[...,1]] - geometry[topology[...,0]]


'''
One common choice of basis(enumerated phi_i) is the piecewise linear.
Over simplex_i: (note phi_i behaves like phi_(i+1) for simplex_(i-1))
    phi_i = (x_(i+1) - x) / (x_(i+1) - x_i) -> @(x_i) = 1, @(x_(i+1)) = 0
    phi_(i+1) = (x - x_i) / (x_(i+1) - x_i) -> @(x_i) = 0, @(x_(i+1)) = 1
The construction will always have n+1 vertices and n faces(assuming conforming triangulation).

This construction allows for a beautiful barycentric coordinate description:
    for any simplex we have to interaction between phi_i and phi_(i+1).
    phi_i + phi_(i+1) = 1, so lets write them in terms of some new coordinate frame lambda_i.
    lambda_i = length_enclosed_by_replacing_x_i_with_x(x) / length(simplex_i)
It turns out this is a general formulation that can be abstracted into any dimension
with the correct notion of Lebesgue measure.

Furthermore, we can now view the del phi_i as del lambda_i:
    del lambda_i = -1 / length(simplex_i)
    del lambda_(i+1) = 1 / length(simplex_i)
So when we integrate over the simplex the cross terms become:
    i,i+1 = -1 / length(simplex_i)
The diagonal terms becomes:
    i,i or i+1,i+1 = 1 / length(simplex_i)

'''
def construct_local_1_1_operator_1d(
    jacobian: torch.Tensor,
) -> torch.Tensor:

    length = torch.abs(jacobian)

    return jacobian.new_tensor([
        [1.0, -1.0],
        [-1.0, 1.0],
    ]) / length
def construct_global_1_1_operator_1d(
    jacobians: torch.Tensor,
    topology: torch.Tensor,
) -> torch.Tensor:

    assert jacobians.numel() == topology.shape[0]

    num_vertices = topology.shape[0] + 1

    A = jacobians.new_zeros(
        (num_vertices, num_vertices)
    )

    for idx, simplex in enumerate(topology):
        local_1_1 = construct_local_1_1_operator_1d(
            jacobians[idx]
        )

        A[simplex[0], simplex[0]] += local_1_1[0, 0]
        A[simplex[1], simplex[0]] += local_1_1[1, 0]
        A[simplex[0], simplex[1]] += local_1_1[0, 1]
        A[simplex[1], simplex[1]] += local_1_1[1, 1]

    return A
def construct_local_0_0_operator_1d(
    jacobian: torch.Tensor,
) -> torch.Tensor:

    length = torch.abs(jacobian)

    return length * jacobian.new_tensor([
        [2.0, 1.0],
        [1.0, 2.0],
    ]) / 6.0

def construct_global_0_0_operator_1d(
    jacobians: torch.Tensor,
    topology: torch.Tensor,
) -> torch.Tensor:

    assert jacobians.numel() == topology.shape[0]

    num_vertices = topology.shape[0] + 1

    M = jacobians.new_zeros(
        (num_vertices, num_vertices)
    )

    for idx, simplex in enumerate(topology):
        local_0_0 = construct_local_0_0_operator_1d(
            jacobians[idx]
        )

        M[simplex[0], simplex[0]] += local_0_0[0, 0]
        M[simplex[1], simplex[0]] += local_0_0[1, 0]
        M[simplex[0], simplex[1]] += local_0_0[0, 1]
        M[simplex[1], simplex[1]] += local_0_0[1, 1]

    return M

 
