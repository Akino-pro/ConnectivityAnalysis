import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi


# -------------------------------------------------------------------------
# 1) Generate sample points (Fibonacci Sphere approach)
# -------------------------------------------------------------------------
def fibonacci_sphere_angles(n):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)
    phi_angles = np.arcsin(z)  # in [-π/2, π/2]
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)
    return theta_angles, phi_angles


n_points = 100
theta, phi = fibonacci_sphere_angles(n_points)

R_outer = 6 * np.pi  # The outer 'reference' radius

x = R_outer * np.cos(theta) * np.cos(phi)
y = R_outer * np.sin(theta) * np.cos(phi)
z = R_outer * np.sin(phi)
points = np.column_stack((x, y, z))

# -------------------------------------------------------------------------
# 2) Compute Spherical Voronoi at radius = 6π
# -------------------------------------------------------------------------
sv = SphericalVoronoi(points, radius=R_outer, center=[0, 0, 0])
sv.sort_vertices_of_regions()

# -------------------------------------------------------------------------
# 3) Provide intervals for each region
#    intervals[i] = a list of [rmin, rmax] pairs for region i
# -------------------------------------------------------------------------
# For demonstration, let's define a simple example:
#   - Region 0 has intervals: [[1.5π, 2π], [3π, 4π]]
#   - Region 1 has intervals: [[2π, 3π]]
#   - All others have no intervals -> no wedge drawn
#
# In a real scenario, you'd provide this list from your data.
# Just ensure 'intervals' is length = n_points, where each element is
# a list of [r_min, r_max] pairs.

intervals = [[] for _ in range(n_points)]  # start empty

# Example: region 0 -> two intervals
intervals[0] = [
    [1.5 * np.pi, 2.0 * np.pi],
    [3.0 * np.pi, 4.0 * np.pi]
]

# Example: region 1 -> one interval
intervals[1] = [
    [1.8 * np.pi, 3.5 * np.pi]
]

intervals[2] = [[0, 6 * np.pi]]


# You could fill in other intervals as needed.

# -------------------------------------------------------------------------
# 4) Helper function: build truncated wedge from a single [rmin, rmax]
# -------------------------------------------------------------------------
def build_truncated_wedge(region_vertices, rmin, rmax, R_outer):
    """
    Given 'region_vertices' (each on radius R_outer),
    build triangular faces for the wedge from rmin to rmax.
    Returns a list of triangular faces (each face is [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)]).
    """
    # Scale the polygon from radius R_outer to rmin and rmax
    poly_rmin = region_vertices * (rmin / R_outer)
    poly_rmax = region_vertices * (rmax / R_outer)

    faces = []

    # --- Top Face (at rmax) ---
    centroid_rmax = np.mean(poly_rmax, axis=0)
    for i in range(len(poly_rmax)):
        v1 = poly_rmax[i]
        v2 = poly_rmax[(i + 1) % len(poly_rmax)]
        faces.append([centroid_rmax, v1, v2])

    # --- Bottom Face (at rmin) ---
    # Only if rmin > 0 (otherwise, it collapses to the origin)
    if rmin > 0:
        centroid_rmin = np.mean(poly_rmin, axis=0)
        for i in range(len(poly_rmin)):
            v1 = poly_rmin[i]
            v2 = poly_rmin[(i + 1) % len(poly_rmin)]
            # Reverse order to keep consistent normal
            faces.append([centroid_rmin, v2, v1])

    # --- Side Faces ---
    for i in range(len(region_vertices)):
        i_next = (i + 1) % len(region_vertices)
        vminA = poly_rmin[i]
        vminB = poly_rmin[i_next]
        vmaxA = poly_rmax[i]
        vmaxB = poly_rmax[i_next]
        # Split quadrilateral into two triangles
        faces.append([vminA, vminB, vmaxB])
        faces.append([vminA, vmaxB, vmaxA])

    return faces


# -------------------------------------------------------------------------
# 5) Plot each region's truncated wedges
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, region in enumerate(sv.regions):
    region_vertices = sv.vertices[region]  # Nx3

    # intervals[i] is a list of [rmin, rmax] pairs
    for [rmin, rmax] in intervals[i]:
        wedge_triangles = build_truncated_wedge(
            region_vertices, rmin, rmax, R_outer
        )
        wedge_poly = Poly3DCollection(
            wedge_triangles,
            alpha=1.0,  # Adjust transparency
            edgecolor='k',
            facecolor='b'  # Random color for demonstration
        )
        ax.add_collection3d(wedge_poly)

# Optionally, scatter the original sample points
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c='r', s=40, label='Sampled Points')

# Adjust axes
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Multiple Truncated Wedges per Region')

plt.legend()
plt.show()
