import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def fibonacci_sphere_angles(n):
    """
    Generate spherical coordinates (theta, phi) for points evenly distributed on a sphere using the Fibonacci Sphere Algorithm.

    Args:
    - n (int): Number of points.

    Returns:
    - angles (list of tuples): List of (theta, phi) in radians.
        - theta: Azimuthal angle (0 to 2π).
        - phi: Polar angle (-π/2 to π/2).
    """
    angles = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    for i in range(n):
        z = 1 - (2 * i) / (n - 1)  # z ranges from 1 to -1
        phi_angle = np.arcsin(z)  # Polar angle (-π/2 to π/2)
        theta_angle = 2 * np.pi * i / phi  # Azimuthal angle (0 to 2π)
        angles.append((theta_angle % (2 * np.pi), phi_angle))

    return angles


def spherical_to_cartesian(radius, angles):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
    - radius (float): Radius of the sphere.
    - angles (list of tuples): List of (theta, phi) in radians.

    Returns:
    - ndarray: Array of Cartesian coordinates (x, y, z).
    """
    cartesian_coords = []
    for theta, phi in angles:
        x = radius * np.cos(phi) * np.cos(theta)
        y = radius * np.cos(phi) * np.sin(theta)
        z = radius * np.sin(phi)
        cartesian_coords.append([x, y, z])
    return np.array(cartesian_coords)


def find_neighbors_with_threshold(points, k=6, threshold=0.1):
    """
    Find neighbors of each point within a sphere, considering a distance threshold.

    Parameters:
        points (ndarray): Array of shape (N, 3) representing points on the sphere.
        k (int): Number of nearest neighbors to include initially.
        threshold (float): Distance threshold to include additional neighbors.

    Returns:
        list: A list of arrays where each array contains the indices of the neighbors for a point.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)  # k+1 includes the point itself

    neighbors = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # Start with k nearest neighbors
        valid_neighbors = idx[1:]  # Exclude the point itself (idx[0] is the point itself)

        # Find additional neighbors within the distance threshold
        additional_neighbors = tree.query_ball_point(points[i], r=dist[-1] + threshold)

        # Combine and deduplicate neighbors
        combined_neighbors = np.unique(np.concatenate((valid_neighbors, additional_neighbors)))
        combined_neighbors = combined_neighbors[combined_neighbors != i]  # Exclude the point itself
        neighbors.append(combined_neighbors)
    return neighbors


def create_polyhedra_with_closest_midpoints(points, neighbors, m=3):
    """
    Create polyhedra for each point on the sphere, connecting midpoints to their closest neighbors.

    Parameters:
        points (ndarray): Array of shape (N, 3) representing points on the sphere.
        neighbors (list): List of arrays where each array contains the indices of neighbors for a point.
        m (int): Number of closest midpoint connections to retain.

    Returns:
        list: A list of polyhedra (vertices and facets).
    """
    polyhedra = []

    for i, point in enumerate(points):
        midpoints = []
        for j in neighbors[i]:
            midpoint = (point + points[j]) / 2
            midpoints.append(midpoint)
        midpoints = np.array(midpoints)

        # Find closest connections between midpoints
        dist_matrix = distance.cdist(midpoints, midpoints)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-connections

        closest_connections = []
        for idx, row in enumerate(dist_matrix):
            closest_indices = np.argsort(row)[:m]  # Take m closest neighbors
            for ci in closest_indices:
                if {idx, ci} not in closest_connections:  # Ensure undirected uniqueness
                    closest_connections.append({idx, ci})

        # Construct facets using closest connections
        center = [0, 0, 0]  # Sphere's center
        facets = []
        for connection in closest_connections:
            idx1, idx2 = list(connection)
            facets.append([midpoints[idx1], midpoints[idx2], center])

        polyhedra.append({"vertices": np.vstack((midpoints, center)), "facets": facets})

    return polyhedra


def plot_polyhedra(ax, polyhedra, indices, points):
    """
    Plot specified polyhedra based on given indices.

    Parameters:
        ax: Matplotlib 3D axis.
        polyhedra: List of polyhedra data (vertices and facets).
        indices: List of indices for polyhedra to plot.
        points: Array of sampled points on the sphere.
    """
    for i in indices:
        polyhedron = polyhedra[i]
        facets = polyhedron["facets"]

        # Plot each facet
        poly3d = Poly3DCollection(facets, alpha=0.5, edgecolor='k')
        ax.add_collection3d(poly3d)

    # Plot sampled points as red dots
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=20)


# Main Execution
radius = 2 * np.pi
num_points = 100
angles = fibonacci_sphere_angles(num_points)
points = spherical_to_cartesian(radius, angles)
neighbors = find_neighbors_with_threshold(points, k=4, threshold=0.1)
polyhedra = create_polyhedra_with_closest_midpoints(points, neighbors, m=2)

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plot_indices = [0]  # Specify indices of polyhedra to visualize
plot_polyhedra(ax, polyhedra, plot_indices, points)

# Plot Settings
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])
ax.set_box_aspect([1, 1, 1])
plt.show()
