import heapq

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
import trimesh

step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size


def union_ranges(ranges):
    """
    Computes the union of a list of ranges, with special handling for ranges that wrap around
    [-π, π]. If such ranges are detected, they are replaced by modified ranges.

    Parameters:
        ranges (list of tuples): List of ranges, where each range is represented as (a, b).

    Returns:
        list of tuples: The merged union of ranges as a list of non-overlapping ranges.
    """
    if not ranges:
        return []

    # Optional: Validate input ranges
    for a, b in ranges:
        if a > b:
            raise ValueError(f"Invalid range: ({a}, {b}). Start must be <= end.")

    # Sort the ranges by their starting point (and by end if start points are the same)
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

    # Initialize the merged ranges with the first range
    merged = [ranges[0]]
    merge_threshold = terminate_threshold
    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1] or (current[0] - last[1] <= merge_threshold):  # Handle small gaps
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged


def plot_shifted_arcs_on_sphere(angles, theta_ranges_per_point, samples_per_arc=50):
    """
    Plot arcs on the unit sphere by 'shifting' each point's theta over
    one or more intervals. The phi (colatitude) for each point remains fixed.

    Args:
        angles (list or np.ndarray): shape (n, 2), each entry is (theta_i, phi_i).
                                     - theta_i in [0, 2π)
                                     - phi_i in [0, π]
        theta_ranges_per_point (list of list of tuples): shape (n, ), where each element
            is a list of intervals [(tmin1, tmax1), (tmin2, tmax2), ...].
            Example: [
                [(0,1), (2,3)],  # intervals for point #1
                [(0, 2*np.pi)],  # intervals for point #2
                ...
            ]
        samples_per_arc (int): number of sample points when drawing each arc.

    Returns:
        None. Displays a 3D matplotlib figure with the arcs plotted on a unit sphere.
    """
    angles = np.asarray(angles)
    n_points = angles.shape[0]

    if len(theta_ranges_per_point) != n_points:
        raise ValueError("Length of theta_ranges_per_point must match number of angles.")

    # 1) Create a reference sphere mesh
    phi_s = np.linspace(0, np.pi, 40)
    theta_s = np.linspace(0, 2 * np.pi, 40)
    phi_s, theta_s = np.meshgrid(phi_s, theta_s)
    x_s = np.sin(phi_s) * np.cos(theta_s)
    y_s = np.sin(phi_s) * np.sin(theta_s)
    z_s = np.cos(phi_s)

    # 2) Prepare the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.2, linewidth=0)

    # 3) Plot arcs for each point
    # colors = plt.cm.rainbow(np.linspace(0, 1, n_points))  # for variety
    for i in range(n_points):
        theta_i, phi_i = angles[i]
        # We'll also plot the original point as a reference
        # x0, y0, z0 = spherical_to_cartesian(theta_i, phi_i)
        # ax.scatter([x0], [y0], [z0], color='b', s=40)

        intervals = theta_ranges_per_point[i]
        for (tmin, tmax) in intervals:
            # Sample points along [tmin, tmax]
            tvals = np.linspace(tmin, tmax, samples_per_arc)
            arc_x, arc_y, arc_z = [], [], []
            for t in tvals:
                xA, yA, zA = spherical_to_cartesian(t, phi_i)
                arc_x.append(xA)
                arc_y.append(yA)
                arc_z.append(zA)

            # Plot the arc
            ax.plot(arc_x, arc_y, arc_z, color='b', lw=6)

    # 4) Final touches
    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1]);
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Shifted Arcs on Unit Sphere")
    plt.show()


def fibonacci_sphere_angles(n):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    z = np.linspace(1, -1, n)
    phi_angles = np.arcsin(z)  # in [-π/2, π/2]
    theta_angles = (2 * np.pi * np.arange(n) / phi) % (2 * np.pi)
    result = []
    for i in range(n):
        result.append([theta_angles[i], phi_angles[i]])
    return result


def build_truncated_wedge(region_vertices, rmin, rmax, R_outer=1.0):
    """
    Given 'region_vertices' on a sphere of radius R_outer,
    build triangular faces for the wedge from r = rmin to r = rmax.
    Returns a list of triangular faces, each face is [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
    """
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
    if rmin > 0:
        centroid_rmin = np.mean(poly_rmin, axis=0)
        for i in range(len(poly_rmin)):
            v1 = poly_rmin[i]
            v2 = poly_rmin[(i + 1) % len(poly_rmin)]
            # Reverse to keep consistent normals
            faces.append([centroid_rmin, v2, v1])

    # --- Side Faces ---
    for i in range(len(region_vertices)):
        i_next = (i + 1) % len(region_vertices)
        vminA = poly_rmin[i]
        vminB = poly_rmin[i_next]
        vmaxA = poly_rmax[i]
        vmaxB = poly_rmax[i_next]
        # Two triangles for the side quadrilateral
        faces.append([vminA, vminB, vmaxB])
        faces.append([vminA, vmaxB, vmaxA])

    return faces


def find_voronoi_region(sv, p):
    """
    Find which Voronoi region (index) the point p (on the unit sphere)
    belongs to, by checking which generator in 'sv.points' is closest.

    Inputs:
        sv: SphericalVoronoi object
        p:  A 3D point on the sphere (x, y, z) ~ unit radius

    Returns:
        region_index: the index of the region in sv.regions to which p belongs.
                      This is effectively the index of the closest generator.
    """
    # Brute force: find which generator is closest in Euclidean distance
    # (for n=100 or so, this is perfectly fast enough.)
    points = sv.points  # shape (n, 3)
    # Distances from p to each generator
    dists = np.linalg.norm(points - p, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx


def plot_extruded_regions_covered_by_arcs(theta_phi_list,
                                          theta_ranges_all,
                                          samples_per_arc=50,
                                          extrude_radius=2 * np.pi,
                                          facecolor='b',
                                          alpha=1.0):
    """
    1) Build SphericalVoronoi from 'theta_phi_list' (points on the unit sphere).
    2) For each point i, we have arcs in theta_ranges_all[i].
       - Each arc is (tmin, tmax) in theta.
       - We keep phi_i fixed, vary theta in [tmin, tmax].
       - For each sample on that arc, find the Voronoi region index it belongs to.
    3) Collect the union of all region indices visited by all arcs.
    4) Plot only those visited regions, extruded from r=0 to r=extrude_radius.

    Args:
        theta_phi_list   : list of (theta, phi) for each generator on unit sphere
        theta_ranges_all : list of lists of intervals [(tmin, tmax), ...] for each generator
        samples_per_arc  : number of sample points per arc
        extrude_radius   : how far to extrude each region, e.g. 2π
        facecolor, alpha : polygon rendering style

    Returns:
        None (displays the 3D figure)
    """
    # 1) Convert angles to 3D points on the sphere
    points = []
    for (theta, phi) in theta_phi_list:
        x, y, z = spherical_to_cartesian(theta, phi)
        points.append([x, y, z])
    points = np.array(points)

    # 2) Build spherical Voronoi
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # We'll keep a set of region indices that are "visited" by any arc
    visited_region_indices = set()

    # 3) For each point i, arcs in theta_ranges_all[i]
    for i, intervals in enumerate(theta_ranges_all):
        # If no arcs are given for this point, skip
        if not intervals:
            continue

        # We'll use the phi_i from the generator
        theta_i, phi_i = theta_phi_list[i]

        for (tmin, tmax) in intervals:
            # sample the arc in small steps
            thetas = np.linspace(tmin, tmax, samples_per_arc)

            for t in thetas:
                # (x_arc, y_arc, z_arc) on the sphere
                x_arc, y_arc, z_arc = spherical_to_cartesian(t, phi_i)
                p_arc = np.array([x_arc, y_arc, z_arc])
                # find which region p_arc belongs to
                r_idx = find_voronoi_region(sv, p_arc)
                visited_region_indices.add(r_idx)

    # 4) Plot only visited regions, extruded out to radius=extrude_radius
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for r_idx in visited_region_indices:
        region = sv.regions[r_idx]
        region_vertices = sv.vertices[region]  # Nx3 on the unit sphere

        # Here we extrude from r=0 to r=extrude_radius
        wedge_faces = build_truncated_wedge(region_vertices,
                                            rmin=0,
                                            rmax=extrude_radius,
                                            R_outer=1.0)
        wedge_poly = Poly3DCollection(
            wedge_faces,
            alpha=alpha,
            edgecolor='b',
            facecolor=facecolor
        )
        ax.add_collection3d(wedge_poly)

    # Optionally, show the original sphere points
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, c='r', label='Generators')

    # Format the axes
    ax.set_box_aspect((1, 1, 1))
    # If extrude_radius=2π, let's keep them symmetrical
    R = extrude_radius
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Voronoi Regions Covered by Arcs, Extruded to r={extrude_radius}")
    plt.show()
    return ax


def spherical_to_cartesian(theta, phi):
    """
    Using the 'latitude' convention:
       theta: azimuth in [0, 2π)
       phi:   latitude in [-π/2, π/2]
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    return x, y, z


def build_truncated_wedge(region_vertices, rmin, rmax, R_outer=1.0):
    """
    Build triangular faces that form a 'wedge' from r=rmin to r=rmax
    around the polygon given by region_vertices (on sphere of radius=R_outer).
    Returns a list of triangular faces: each face is [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
    """
    poly_rmin = region_vertices * (rmin / R_outer)
    poly_rmax = region_vertices * (rmax / R_outer)

    faces = []

    # Top Face (at rmax)
    centroid_rmax = np.mean(poly_rmax, axis=0)
    for i in range(len(poly_rmax)):
        v1 = poly_rmax[i]
        v2 = poly_rmax[(i + 1) % len(poly_rmax)]
        faces.append([centroid_rmax, v1, v2])

    # Bottom Face (at rmin)
    if rmin > 0:
        centroid_rmin = np.mean(poly_rmin, axis=0)
        for i in range(len(poly_rmin)):
            v1 = poly_rmin[i]
            v2 = poly_rmin[(i + 1) % len(poly_rmin)]
            # Reverse to keep consistent normals
            faces.append([centroid_rmin, v2, v1])

    # Side Faces
    for i in range(len(region_vertices)):
        i_next = (i + 1) % len(region_vertices)
        vminA = poly_rmin[i]
        vminB = poly_rmin[i_next]
        vmaxA = poly_rmax[i]
        vmaxB = poly_rmax[i_next]
        # Two triangles for the side quadrilateral
        faces.append([vminA, vminB, vmaxB])
        faces.append([vminA, vmaxB, vmaxA])

    return faces


def get_extruded_wedges(theta_phi_list,
                        theta_ranges_all,
                        alpha_lists,
                        samples_per_arc=50,
                        extrude_radius=2 * np.pi,
                        do_plot=False,
                        color='b'):
    """
    Computes the extruded wedges for all arcs and returns them as a single list of triangular faces.
    If `do_plot` is True, also plots these wedges in a 3D figure.

    Returns:
        all_wedge_faces : list of triangles (each is 3 points in [x, y, z])
    """
    # 1) Convert angles to 3D points on the sphere
    points = []
    for (theta, phi) in theta_phi_list:
        x, y, z = spherical_to_cartesian(theta, phi)
        points.append([x, y, z])
    points = np.array(points)

    # 2) Build spherical Voronoi
    sv = SphericalVoronoi(points, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    # We'll keep a set of region indices that are visited by any arc
    visited_region_indices = set()

    # 3) For each point i, arcs in theta_ranges_all[i]
    for i, intervals in enumerate(theta_ranges_all):
        if not intervals:
            continue
        # We'll use phi_i from the i-th point
        theta_i, phi_i = theta_phi_list[i]

        for (tmin, tmax) in intervals:
            # Sample points along the arc in theta
            thetas = np.linspace(tmin, tmax, samples_per_arc)
            # Track visited regions for this arc so we only union once per region
            visited_for_this_arc = set()

            for t in thetas:
                x_arc, y_arc, z_arc = spherical_to_cartesian(t, phi_i)
                p_arc = np.array([x_arc, y_arc, z_arc])
                # Find which Voronoi region p_arc belongs to
                r_idx = find_voronoi_region(sv, p_arc)
                visited_for_this_arc.add(r_idx)

            # Union alpha_lists[r_idx] with alpha_lists[i] for each visited region
            for r_idx in visited_for_this_arc:
                visited_region_indices.add(r_idx)
                # Concatenate existing intervals from alpha_lists[r_idx] and alpha_lists[i]
                combined = alpha_lists[r_idx] + alpha_lists[i]
                # Use the union_ranges(...) function to merge them
                alpha_lists[r_idx] = union_ranges(combined)

    # 4) Build the wedge faces for each visited region
    all_wedge_faces = []
    for r_idx in visited_region_indices:
        region = sv.regions[r_idx]
        region_vertices = sv.vertices[region]  # Nx3 on the unit sphere

        wedge_faces = build_truncated_wedge(region_vertices,
                                            rmin=0,
                                            rmax=extrude_radius,
                                            R_outer=1.0)
        all_wedge_faces.extend(wedge_faces)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Add polygon collections
        wedge_poly = Poly3DCollection(all_wedge_faces,
                                      alpha=1.0,
                                      edgecolor='k',
                                      facecolor=color)
        ax.add_collection3d(wedge_poly)

        # Format the axes
        R = extrude_radius
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim([-R, R])
        ax.set_ylim([-R, R])
        ax.set_zlim([-R, R])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Voronoi Regions Covered by Arcs, Extruded to r={extrude_radius}")
        plt.show()
    return all_wedge_faces, alpha_lists


def wedge_faces_to_binary_volume(all_wedge_faces, NX=50, NY=50, NZ=50):
    """
    Given a list of triangular wedge faces (each face is [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]),
    build a single 3D mesh with trimesh, then produce an (NX, NY, NZ) occupancy grid.

    Returns:
        binary_volume: np.ndarray of shape (NX, NY, NZ), dtype=uint8
        xs, ys, zs   : 1D coordinate arrays for each axis
    """
    unique_vertices = []
    vertex_map = {}
    faces = []

    for tri in all_wedge_faces:
        face_indices = []
        for v in tri:
            v_tuple = (v[0], v[1], v[2])
            if v_tuple not in vertex_map:
                vertex_map[v_tuple] = len(unique_vertices)
                unique_vertices.append(v_tuple)
            face_indices.append(vertex_map[v_tuple])
        faces.append(face_indices)

    vertices = np.array(unique_vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    # Build and clean the trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # Build axis-aligned 3D grid from bounding box
    bounds = mesh.bounds  # shape (2,3)
    min_corner = bounds[0]
    max_corner = bounds[1]

    xs = np.linspace(min_corner[0], max_corner[0], NX)
    ys = np.linspace(min_corner[1], max_corner[1], NY)
    zs = np.linspace(min_corner[2], max_corner[2], NZ)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='xy')
    points_to_check = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    # Check which points are inside the mesh
    inside_mask = mesh.contains(points_to_check)
    inside_mask_3d = inside_mask.reshape((NX, NY, NZ))

    # Convert to a binary occupancy grid
    binary_volume = inside_mask_3d.astype(np.uint8)

    return binary_volume


def track_top_5():
    top_5 = []  # Min heap to keep track of top-5 numbers with indices and properties

    def update(num, index, property_value1, property_value2):
        if len(top_5) < 5:
            heapq.heappush(top_5, (num, index, property_value1, property_value2))
        else:
            heapq.heappushpop(top_5, (num, index, property_value1, property_value2))  # Maintain only top-5

    def get_top_5():
        return sorted(top_5, key=lambda x: x[0], reverse=True)  # Return sorted top-5 list

    return update, get_top_5


def normalize_and_map_colors(values, cmap_name='rainbow'):
    """
    Normalizes a list of values to the range [0, 1] and maps them to colors from a given colormap.

    Parameters:
        values (list or np.array): List of numerical values.
        cmap_name (str): Name of the colormap to use (default is 'viridis').

    Returns:
        list: List of RGB color tuples corresponding to the normalized values.
        matplotlib.cm.ScalarMappable: A colormap mappable for the colorbar.
    """
    values = np.array(values)  # Convert to NumPy array
    min_val, max_val = np.min(values), np.max(values)

    # Normalize values to [0,1] range
    if max_val - min_val > 0:
        normalized_values = (values - min_val) / (max_val - min_val)
    else:
        normalized_values = np.zeros_like(values)  # If all values are the same, use zero.

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Map normalized values to colors
    colors = [cmap(val) for val in normalized_values]

    # Create a mappable object for colorbar
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar to work properly

    return colors, sm
"""
### Example Usage:
values = [10, 20, 30, 40, 50, 60, 70]
colors, sm = normalize_and_map_colors(values)

fig, ax = plt.subplots()
sc = ax.scatter(values, np.zeros_like(values), c=colors, s=100)
cbar = plt.colorbar(sm, ax=ax, label='Value Spectrum')  # Ensure colorbar is linked to the mappable

plt.show()
"""
