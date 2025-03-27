import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
from helper_functions import fibonacci_sphere_angles


def spherical_to_cartesian(theta, phi):
    """Convert spherical (theta, phi) to Cartesian (x, y, z) on a unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def generate_circle(center, normal, radius=0.1, num_points=100):
    """
    Generate a full circle centered at `center` and lying in a plane perpendicular to `normal`.

    Args:
        center (np.array): 3D coordinates of the center.
        normal (np.array): 3D normal vector defining the plane of the circle.
        radius (float): Radius of the circle.
        num_points (int): Number of points to define the circle.

    Returns:
        np.array: (num_points, 3) array of circle points in 3D space.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Find a perpendicular vector to define the circle plane
    perp_vector = np.cross(normal, np.array([1, 0, 0]))
    if np.linalg.norm(perp_vector) < 1e-6:  # Handle edge case if normal is colinear with x-axis
        perp_vector = np.cross(normal, np.array([0, 1, 0]))
    perp_vector = perp_vector / np.linalg.norm(perp_vector)

    # Create a second perpendicular vector
    second_vector = np.cross(normal, perp_vector)

    # Generate points along the circle with the given radius
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([
        center + radius * (np.cos(angle) * perp_vector + np.sin(angle) * second_vector)
        for angle in angles
    ])

    return circle_points



def plot_voronoi_regions_on_sphere(theta_phi_list, selected_indices, angular_ranges):
    """
    Plot spherical Voronoi regions, coloring only selected regions and drawing modified circles.

    Args:
        theta_phi_list (list of (float, float)):
            Spherical coordinates (theta, phi) for each generator (radians).
        selected_indices (list of int):
            Indices of the 16 selected orientations to be colored.
        angular_ranges (list of tuple):
            List of 16 (min_angle, max_angle) defining the cut sections of circles.

    Returns:
        ax (matplotlib.axes._subplots.Axes3DSubplot):
            The Matplotlib 3D axes object.
    """

    # Convert generator points to 3D Cartesian coordinates
    points = np.array([spherical_to_cartesian(theta, phi) for theta, phi in theta_phi_list])

    # Compute Spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # Compute depth for each region (Z-value of centroid)
    region_depths = []
    for idx, region in enumerate(sv.regions):
        region_vertices = sv.vertices[region]
        centroid = np.mean(region_vertices, axis=0)
        region_depths.append((centroid[2], region_vertices, idx))  # Store depth, vertices, and index

    # Sort regions by depth (back to front)
    region_depths.sort(reverse=True, key=lambda x: x[0])

    # Plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Voronoi regions, only coloring the selected ones
    for _, region_vertices, idx in region_depths:
        face_color = 'lightgreen' if idx in selected_indices else (1, 1, 1, 0)  # Fully transparent for others
        poly = Poly3DCollection([region_vertices], facecolor=face_color, edgecolor='k', alpha=1 if idx in selected_indices else 0)
        ax.add_collection3d(poly)

    # Mark the centers (Voronoi generators) with green dots
    selected_points = points[selected_indices]  # Extract only the selected points
    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2],
               color='green', s=50, edgecolors='black',
               depthshade=False, zorder=10, label="Grid centers with non-empty alpha range")

    # Draw cut circles for selected orientations
    for i, idx in enumerate(selected_indices):
        center = points[idx]  # Region center
        normal = center  # The normal is aligned with the orientation
        min_angle, max_angle = angular_ranges[i]  # Get the range to cut the circle

        # Generate full circle
        full_circle = generate_circle(center, normal)

        # Select only points within the desired angular range
        angles = np.linspace(0, 2 * np.pi, len(full_circle))
        mask = (angles >= min_angle) & (angles <= max_angle)
        cut_circle = full_circle[mask]

        # Plot the cut circle segment
        ax.plot(cut_circle[:, 0], cut_circle[:, 1], cut_circle[:, 2], color='red', linewidth=2,zorder=100)

    # Set axis limits and labels
    ax.set_box_aspect((1, 1, 1))
    limit = 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    # Show legend
    ax.legend()

    plt.show()
    return ax


# Generate orientation samples
orientation_samples = 64
theta_phi_list = fibonacci_sphere_angles(orientation_samples)

# Randomly select 16 unique orientations
selected_indices = np.random.choice(orientation_samples, 16, replace=False)

min_values = np.random.uniform(-2*np.pi, 0, size=16)  # Lower bound in [-2π, π]
max_values = np.random.uniform(np.pi, 2*np.pi, size=16)  # Ensure at least π separation

# Stack into (16,2) array ensuring min < max
angular_ranges = np.column_stack((min_values, max_values))

# Plot with selected regions colored and cut circles drawn
plot_voronoi_regions_on_sphere(theta_phi_list, selected_indices, angular_ranges)
