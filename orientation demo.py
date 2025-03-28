import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
import random

def spherical_to_cartesian(theta, phi, radius=2*np.pi):
    """ Convert spherical coordinates (theta, phi) to Cartesian (x, y, z). """
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    return np.array([x, y, z])

def fibonacci_sphere_angles(n):
    """ Generate approximately uniform points on a sphere using Fibonacci spiral. """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)
    phi_angles = np.arcsin(z)  # in [-π/2, π/2]
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)
    return list(zip(theta_angles, phi_angles))

def set_axes_equal(ax, radius):
    """ Set equal aspect ratio for 3D plots to ensure true spherical visualization. """
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    limits = [-radius, radius]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)

def plot_voronoi_regions_on_sphere(theta_phi_list, radius=2*np.pi):
    """
    Plot all spherical Voronoi regions with transparency and mark region centers.

    Args:
        theta_phi_list (list of (float, float)): Spherical coordinates (theta, phi) for each generator (radians).
        radius (float): Radius of the sphere.

    Returns:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The Matplotlib 3D axes object.
    """
    # Convert generator points to 3D Cartesian coordinates
    points = np.array([spherical_to_cartesian(theta, phi, radius) for theta, phi in theta_phi_list])

    # Compute Spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius=radius, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # Randomly select 50 orientations to highlight
    num_selected = 50
    selected_indices = set(random.sample(range(len(theta_phi_list)), num_selected))

    # Plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Voronoi regions
    for i, region in enumerate(sv.regions):
        region_vertices = sv.vertices[region]
        alpha_value = 1.0 if i in selected_indices else 0.1  # Fully opaque for selected, otherwise transparent
        poly = Poly3DCollection([region_vertices], facecolor='green', edgecolor='k', alpha=alpha_value)
        ax.add_collection3d(poly)

    # Mark the centers (Voronoi generators) with red dots
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='k', s=10, label="Region Centers")

    # Set equal axis scaling
    set_axes_equal(ax, radius)
    tick_labels = [r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$']
    tick_values = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels, fontsize=18)
    ax.set_zticks(tick_values)
    ax.set_zticklabels(tick_labels, fontsize=18)


    # Show legend
    ax.legend()

    plt.show()
    return ax

# Generate points using Fibonacci sphere sampling
orientation_samples = 200  # 64
theta_phi_list = fibonacci_sphere_angles(orientation_samples)

# Plot with sphere radius of 2π and 50 selected opaque regions
plot_voronoi_regions_on_sphere(theta_phi_list, radius=2*np.pi)

