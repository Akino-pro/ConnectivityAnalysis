import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi

# Function to generate spherical coordinates using the Fibonacci Sphere Algorithm
def fibonacci_sphere_angles(n):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)  # Evenly spaced z values from 1 to -1
    phi_angles = np.arcsin(z)  # Polar angles (-π/2 to π/2)
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)  # Azimuthal angles (0 to 2π)
    return theta_angles, phi_angles

# Generate 10 points on the sphere
n_points = 10
theta, phi = fibonacci_sphere_angles(n_points)

# Convert spherical coordinates to Cartesian with a radius of 6π
radius = 6 * np.pi
x = radius * np.cos(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.cos(phi)
z = radius * np.sin(phi)
points = np.vstack((x, y, z)).T

# Compute Voronoi regions on the sphere
sv = SphericalVoronoi(points, radius=radius, center=[0, 0, 0])
sv.sort_vertices_of_regions()

# Plot Voronoi polyhedra
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Voronoi regions, connecting each vertex to the origin
for region in sv.regions:
    polygon = sv.vertices[region]
    for vertex in polygon:
        # Draw a line connecting each vertex to the origin
        ax.plot([0, vertex[0]], [0, vertex[1]], [0, vertex[2]], 'b-', alpha=0.8)
    # Add polyhedron faces
    poly3d = [[tuple(vertex) for vertex in polygon]]
    ax.add_collection3d(Poly3DCollection(poly3d, alpha=0.5, edgecolor='k'))

# Plot the sampled points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=30, label='Sampled Points')

# Configure plot appearance
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Voronoi Polyhedra on a Sphere with Radius 6π')
plt.legend()
plt.show()
