import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi


###############################################################################
# Utility functions
###############################################################################

def spherical_to_cartesian(theta, phi, radius=1.0):
    """
    Convert spherical coords (theta, phi, radius) to Cartesian (x, y, z).
    theta = angle from z-axis
    phi   = angle in x-y plane
    """
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])


def fibonacci_sphere_angles(samples):
    """
    Approximate uniform distribution on the sphere via Fibonacci spiral.
    Returns a list of (theta, phi) angles.
    """
    indices = np.arange(samples)
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    angles = []
    for i in indices:
        y = 1.0 - (i / float(samples - 1)) * 2.0
        r = np.sqrt(1.0 - y * y)
        # angle from z-axis
        theta = np.arccos(y)
        # distribute around phi
        phi = (2.0 * np.pi * i) / phi_golden
        phi = phi % (2.0 * np.pi)
        angles.append((theta, phi))
    return angles


###############################################################################
# Radial clipping (Sutherland–Hodgman style)
###############################################################################

def clip_polygon_radially(vertices, radius_cut, keep_inside=True):
    """
    Clips a polygon in 3D (array of shape (N,3)) against sphere radius = radius_cut.
    If keep_inside=True, keeps norm <= radius_cut; otherwise keeps norm >= radius_cut.
    """
    if len(vertices) < 3:
        return np.array([])

    clipped = []
    n = len(vertices)
    for i in range(n):
        curr = vertices[i]
        nxt = vertices[(i + 1) % n]
        curr_norm = np.linalg.norm(curr)
        nxt_norm = np.linalg.norm(nxt)

        if keep_inside:
            curr_in = (curr_norm <= radius_cut)
            nxt_in = (nxt_norm <= radius_cut)
        else:
            curr_in = (curr_norm >= radius_cut)
            nxt_in = (nxt_norm >= radius_cut)

        # Keep current if it's "inside"
        if curr_in:
            clipped.append(curr)

        # If crossing boundary, find intersection
        if curr_in != nxt_in:
            edge_dir = nxt - curr
            A = edge_dir.dot(edge_dir)
            B = 2.0 * curr.dot(edge_dir)
            C = curr.dot(curr) - radius_cut ** 2
            disc = B * B - 4 * A * C
            if disc < 0:
                continue
            sqrtD = np.sqrt(disc)
            t_candidates = [(-B + sqrtD) / (2 * A), (-B - sqrtD) / (2 * A)]
            t_valid = [t for t in t_candidates if 0.0 <= t <= 1.0]
            if not t_valid:
                continue
            t = t_valid[0]
            intersect = curr + t * edge_dir
            clipped.append(intersect)

    clipped = np.array(clipped)
    if len(clipped) < 3:
        return clipped

    # Re-sort by angle in xy-plane
    angles = np.arctan2(clipped[:, 1], clipped[:, 0])
    order = np.argsort(angles)
    return clipped[order]


def clip_face_between_radii(face_vertices, r_min, r_max):
    """
    Clip a single face so we only keep the portion with norm in [r_min, r_max].
    1) clip outside r_min (keep outside=False => >= r_min)
    2) clip outside r_max (keep inside=True  => <= r_max)
    """
    # Step 1: remove anything < r_min
    clipped1 = clip_polygon_radially(face_vertices, r_min, keep_inside=False)
    if len(clipped1) < 3:
        return clipped1

    # Step 2: remove anything > r_max
    clipped2 = clip_polygon_radially(clipped1, r_max, keep_inside=True)
    return clipped2


###############################################################################
# Main function
###############################################################################

def plot_single_region_with_truncation_no_axis(theta_phi_list, R=2 * np.pi):
    """
    - Construct Spherical Voronoi on sphere of radius R
    - Select ONE region
    - Form the polyhedron (origin + the region’s vertices)
    - Color the portions in [pi/2, pi], [pi, 3pi/2], and [3pi/2, 2pi] in different transparencies
    - Remove axis lines, ticks, and labels
    - Enlarge the figure
    """
    # 1) Build the spherical Voronoi
    points = np.array([spherical_to_cartesian(t, p, R) for t, p in theta_phi_list])
    sv = SphericalVoronoi(points, radius=R, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # 2) Pick one region
    idx = random.randint(0, len(sv.regions) - 1)
    region = sv.regions[idx]
    region_vertices = sv.vertices[region]
    apex = np.array([0, 0, 0])

    # 3) Construct full "pyramid" faces
    full_shape_faces = [region_vertices]  # base
    n_verts = len(region_vertices)
    for i in range(n_verts):
        v1 = region_vertices[i]
        v2 = region_vertices[(i + 1) % n_verts]
        full_shape_faces.append([apex, v1, v2])

    # 4) Define radial sections for clipping
    clip_ranges = [(np.pi/2, np.pi, 0.3), (np.pi, 3*np.pi/2, 1.0), (3*np.pi/2, 14*np.pi/8, 0.6)]
    clipped_faces_collections = []
    for r_min, r_max, alpha in clip_ranges:
        clipped_faces = []
        for face in full_shape_faces:
            face_array = np.array(face)
            clipped = clip_face_between_radii(face_array, r_min, r_max)
            if len(clipped) >= 3:
                clipped_faces.append(clipped)
        if clipped_faces:
            clipped_faces_collections.append((clipped_faces, alpha))

    # --- NEW: build "cap" faces at r_min and r_max to preserve shape ---
    cap_faces = []
    for r_min, r_max, _ in clip_ranges:
        cap_faces.append(region_vertices * (r_min / R))
        cap_faces.append(region_vertices * (r_max / R))

    ###########################################################################
    # Plot
    ###########################################################################
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Entire shape in translucent green
    all_faces_collection = Poly3DCollection(
        full_shape_faces, facecolor='green', edgecolor='k', alpha=0.1
    )
    ax.add_collection3d(all_faces_collection)

    # Truncated portions
    for clipped_faces, alpha in clipped_faces_collections:
        truncated_collection = Poly3DCollection(clipped_faces, facecolor='green', edgecolor='k', alpha=alpha)
        ax.add_collection3d(truncated_collection)

    # Cap faces
    for cap in cap_faces:
        ax.add_collection3d(Poly3DCollection([cap], facecolor='green', edgecolor='k', alpha=1.0))

    # Mark region center and origin
    region_center = points[idx]
    ax.scatter(*region_center, color='k', s=10, label='Region Center')
    ax.scatter(0, 0, 0, color='black', s=10, label='Origin')
    ax.scatter(*np.mean(region_vertices, axis=0), color='black', s=10, label='Surface Center')  # Added back center point

    # Adjust viewing limits
    lim = R * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Remove axis lines, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_box_aspect([1, 1, 1])
    ax._axis3don = False

    plt.show()



###############################################################################
# Example
###############################################################################
if __name__ == "__main__":
    # Generate sample directions
    samples = 400
    angles_list = fibonacci_sphere_angles(samples)

    # Plot with no axis, no labels, shape enlarged
    plot_single_region_with_truncation_no_axis(angles_list, R=2 * np.pi)
