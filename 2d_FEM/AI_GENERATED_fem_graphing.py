
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_fem_mesh(geometry, triangles, degree=1):
    """Plot a 2D triangular mesh and equally spaced FEM nodes."""

    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer")

    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    vertices = to_numpy(geometry)
    elements = to_numpy(triangles).astype(np.int64)

    # Generate nodes for the requested polynomial degree.
    weights = np.array([
        [i / degree, j / degree, (degree - i - j) / degree]
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ])

    element_nodes = np.einsum(
        "ni,eij->enj", weights, vertices[elements]
    )
    points = element_nodes.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")

    ax.triplot(
        vertices[:, 0], vertices[:, 1],
        triangles=elements,
        color="gray", linewidth=1,
    )
    ax.scatter(
        points[:, 0], points[:, 1],
        s=25, color="royalblue", zorder=3,
    )

    nodes_per_element = (degree + 1) * (degree + 2) // 2
    ax.set(
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title=f"Degree {degree} — {nodes_per_element} nodes per triangle",
    )

    plt.show()
    return element_nodes

def plot_fem_value(geometry, triangles, values, degree=1,
                   resolution=None, cmap="viridis", show_mesh=True):

    def to_numpy(x):
        return (
            x.detach().cpu().numpy()
            if hasattr(x, "detach")
            else np.asarray(x)
        )

    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer")

    vertices = to_numpy(geometry)
    elements = to_numpy(triangles).astype(np.int64)
    u = to_numpy(values)
    nlocal = (degree + 1) * (degree + 2) // 2

    # For degree 1, also accept one value per geometry vertex.
    if degree == 1 and u.shape == (len(vertices),):
        u = u[elements]

    if u.shape != (len(elements), nlocal):
        raise ValueError(
            f"values must have shape ({len(elements)}, {nlocal})"
        )

    n = max(20, 3 * degree) if resolution is None else resolution
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("resolution must be a positive integer")

    bary = np.array([
        [i / n, j / n, (n - i - j) / n]
        for i in range(n + 1)
        for j in range(n + 1 - i)
    ])
    reference = mtri.Triangulation(bary[:, 1], bary[:, 2])

    # Evaluate the polynomial using all FEM node values.
    indices = [
        (i, j, degree - i - j)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
    ]
    basis = np.ones((len(bary), nlocal))

    for a, powers in enumerate(indices):
        for k, power in enumerate(powers):
            for r in range(power):
                basis[:, a] *= (
                    (degree * bary[:, k] - r) / (power - r)
                )

    xy = np.einsum(
        "ni,eij->enj", bary, vertices[elements]
    ).reshape(-1, 2)
    z = (u @ basis.T).reshape(-1)

    offsets = np.arange(len(elements))[:, None, None] * len(bary)
    connectivity = (
        reference.triangles[None, :, :] + offsets
    ).reshape(-1, 3)

    mesh = mtri.Triangulation(
        xy[:, 0], xy[:, 1], triangles=connectivity
    )

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    artist = ax.tripcolor(mesh, z, shading="gouraud", cmap=cmap)

    if show_mesh:
        ax.triplot(
            vertices[:, 0], vertices[:, 1],
            triangles=elements,
            color="black", linewidth=0.5, alpha=0.35,
        )

    fig.colorbar(artist, ax=ax, label="u")
    ax.set(
        aspect="equal", xlabel="x", ylabel="y",
        title=f"FEM solution — degree {degree}",
    )
    plt.show()
    return fig, ax
