import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
from helper_functions import fibonacci_sphere_angles


def spherical_to_cartesian(theta, phi):
    """
    Convert (theta, phi) to Cartesian (x, y, z) on the unit sphere.
    Assuming here:
       - theta in [0, 2π) = azimuth
       - phi in [-π/2, +π/2], i.e. latitude (0 = equator, +π/2 = north pole).
         (If you are using a different convention, adjust accordingly.)
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    return x, y, z


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
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, c='r', label='Generators')

    # Format the axes
    ax.set_box_aspect((1, 1, 1))
    # If extrude_radius=2π, let's keep them symmetrical
    R = extrude_radius
    ax.set_xlim([-R, R]);
    ax.set_ylim([-R, R]);
    ax.set_zlim([-R, R])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Voronoi Regions Covered by Arcs, Extruded to r={extrude_radius}")
    #plt.legend()
    plt.show()

orientation_samples = 100
theta_phi_list = fibonacci_sphere_angles(orientation_samples)

theta_ranges_all = [[] for _ in range(orientation_samples)]
theta_ranges_all[0] = [(0, 1), (2, 3)]
theta_ranges_all[1] = [(1.5, 1.8)]
theta_ranges_all[2] = [(4.5, 5.5), (5.9, 6.2)]
theta_ranges_all[3] = [(0, 2*np.pi)]

plot_extruded_regions_covered_by_arcs(
    theta_phi_list,
    theta_ranges_all,
    samples_per_arc=40,
    extrude_radius=2*np.pi,
    facecolor='b',
    alpha=1.0
)
