import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Function to generate the corner points of a square grid in x-z plane
def generate_square_grid(n_x, n_z, x_range, z_range):
    """Generate a grid of squares in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    squares = []

    for i in range(n_x):
        for j in range(n_z):
            # Define four corners of the square
            bottom_left = [x_vals[i], 0, z_vals[j]]
            bottom_right = [x_vals[i + 1], 0, z_vals[j]]
            top_right = [x_vals[i + 1], 0, z_vals[j + 1]]
            top_left = [x_vals[i], 0, z_vals[j + 1]]
            squares.append([bottom_left, bottom_right, top_right, top_left])

    return squares


def generate_square_grid_centers(n_x, n_z, x_range, z_range):
    """Generate grid of square centers in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    centers = []

    for i in range(n_x):
        for j in range(n_z):
            center_x = (x_vals[i] + x_vals[i + 1]) / 2
            center_z = (z_vals[j] + z_vals[j + 1]) / 2
            centers.append([center_x, 0, center_z])

    return centers


def generate_2D_square_grid(n_x, n_z, x_range, z_range):
    """Generate a grid of squares in the x-z plane."""
    x_vals = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals = np.linspace(z_range[0], z_range[1], n_z + 1)
    squares = []

    for i in range(n_x):
        for j in range(n_z):
            # Define four corners of the square
            bottom_left = [x_vals[i], z_vals[j]]
            bottom_right = [x_vals[i + 1], z_vals[j]]
            top_right = [x_vals[i + 1], z_vals[j + 1]]
            top_left = [x_vals[i], z_vals[j + 1]]
            squares.append([bottom_left, bottom_right, top_right, top_left])

    return squares


# Function to rotate a set of vertices around the Z-axis
def rotate_around_z(vertices, angle):
    """Rotate vertices around the Z-axis by 'angle' (in radians)."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    return np.dot(vertices, rotation_matrix.T)


# Function to draw an arc by rotating a grid square around the Z-axis
def draw_rotated_grid(ax, square, angle_range, color):
    intersected_lower = angle_range[0]
    intersected_upper = np.min([angle_range[1], np.pi / 2])
    if angle_range[0] >= np.pi / 2: return
    # if square[0][2] >= 0:
    #    if angle_range[0] >= 0 or angle_range[1] <= -np.pi / 2: return
    #    intersected_lower = np.max([angle_range[0], -np.pi / 2])
    #    intersected_upper = np.min([angle_range[1], 0])

    angles = np.linspace(intersected_lower, intersected_upper, 270)

    # Loop through each angle to rotate the square around the Z-axis
    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)

        # Create the rotated face (polygon) for the square
        face = [rotated_vertices[j] for j in range(4)]

        # Add the polygon to the plot with a single color
        ax.add_collection3d(Poly3DCollection([face], color="#E0FFFF", edgecolor="#00008B", linewidth=0.1))


def draw_rotated_grid_by_range_only(ax, square, angle_range, color):
    intersected_lower = angle_range[0]
    intersected_upper = angle_range[1]
    angles = np.linspace(intersected_lower, intersected_upper, int((angle_range[1]-angle_range[0])/(np.pi/180))*3)

    # Loop through each angle to rotate the square around the Z-axis
    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)

        # Create the rotated face (polygon) for the square
        face = [rotated_vertices[j] for j in range(4)]

        # Add the polygon to the plot with a single color
        ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5))

def draw_rotated_grid_by_range_only_transparent(ax, square, angle_range, color, center):
    intersected_lower = angle_range[0]
    intersected_upper = angle_range[1]

    angles = np.linspace(intersected_lower, intersected_upper, int((angle_range[1]-angle_range[0])/(np.pi/180))*3)
    # Loop through each angle to rotate the square around the Z-axis

    rotated_vertices = rotate_around_z(np.array(square), intersected_lower)
    face = [rotated_vertices[j] for j in range(4)]
    ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5,alpha=0.2))
    rotated_vertices = rotate_around_z(np.array(square), intersected_upper)
    face = [rotated_vertices[j] for j in range(4)]
    ax.add_collection3d(Poly3DCollection([face], color=color, edgecolor=color, linewidth=0.5,alpha=0.2))

    # Prepare curves for vertices and center
    vertex_curves = [[], [], [], []]
    center_curve = []

    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)
        for idx in range(4):
            vertex_curves[idx].append(rotated_vertices[idx])
        rotated_center = rotate_around_z(np.array([center]), angle)[0]
        center_curve.append(rotated_center)

    # Plot the curves for each vertex
    for vertex_curve in vertex_curves:
        vertex_curve = np.array(vertex_curve)
        ax.plot(vertex_curve[:, 0], vertex_curve[:, 1], vertex_curve[:, 2], color='black', linewidth=0.8)

    # Plot the curve for the center
    center_curve = np.array(center_curve)
    ax.plot(center_curve[:, 0], center_curve[:, 1], center_curve[:, 2], color='black', linewidth=2.0)


def generate_grid_centers(n_x, n_z, num_points, x_range, z_range):
    """Generate the center points of each grid in the x-z plane with correct order."""
    # Ensure that n_x * n_z equals the number of points (or close)
    if n_x * n_z != num_points:
        raise ValueError(f"Number of points ({num_points}) does not match grid layout ({n_x} * {n_z})")

    # Calculate the step size for each square in the grid
    x_step = (x_range[1] - x_range[0]) / n_x
    z_step = (z_range[1] - z_range[0]) / n_z

    # Initialize the list to store centers in the correct order
    centers = []

    # Generate center points for each square by iterating in the same order as `generate_square_grid`
    for i in range(n_x):
        for j in range(n_z):
            # Calculate the center of each square
            x_center = x_range[0] + (i + 0.5) * x_step
            z_center = z_range[0] + (j + 0.5) * z_step
            centers.append([x_center, 0, z_center])  # y = 0 as we're in the x-z plane

    # Convert list to numpy array for compatibility
    return np.array(centers)


def sample_square_points(vertices, num_points=10):
    """Sample points within the square defined by four vertices."""
    points = []
    for i in range(4):
        start, end = vertices[i], vertices[(i + 1) % 4]
        # Interpolate between two vertices to get points along the edge
        edge_points = np.linspace(start, end, num_points)
        points.extend(edge_points)
    return np.array(points)


# Generate scatter points for each rotated square
def generate_rotated_scatter_points(grid_squares, angle_range, num_samples=5):
    """Generate scattered points for each square in grid rotated over specified angle range."""
    scatter_points = []
    angles = np.linspace(angle_range[0], angle_range[1], 60)  # Adjust angle steps as needed

    for square in grid_squares:
        for angle in angles:
            # Rotate square vertices and sample points
            rotated_vertices = rotate_around_z(np.array(square), angle)
            square_points = sample_square_points(rotated_vertices, num_points=num_samples)
            scatter_points.append(square_points)

    # Flatten the list of arrays into a single array of points
    return np.vstack(scatter_points)


"""
def generate_binary_matrix(n_x, n_z, x_range, z_range, grid_size, angle_ranges):
    
    # Generate x and z values for the grid in x-z plane
    x_vals_grid = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals_grid = np.linspace(z_range[0], z_range[1], n_z + 1)

    # Calculate the increments
    delta_x = (x_range[1] - x_range[0]) / n_x
    delta_z = (z_range[1] - z_range[0]) / n_z

    # Create the 3D grid (voxel centers)
    x_vals = np.linspace(-x_range[1], x_range[1], grid_size[0])
    y_vals = np.linspace(-x_range[1], x_range[1], grid_size[1])
    z_vals = np.linspace(z_range[0], z_range[1], grid_size[2])
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    # Compute radial distance and angular coordinate from Z-axis
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)  # Theta ranges from -pi to pi

    # Initialize the binary matrix
    binary_matrix = np.zeros(X.shape, dtype=bool)

    # Check that angle_ranges has the correct length
    if len(angle_ranges) != n_x * n_z:
        raise ValueError(f"angle_ranges should have length {n_x * n_z}, but has length {len(angle_ranges)}")

    # Determine which points are within the rotated grid considering angle ranges per grid square
    square_index = 0
    for i in range(n_x):
        for j in range(n_z):
            # Define the boundaries of the square in x-z plane
            x_min = x_range[0] + i * delta_x
            x_max = x_min + delta_x
            z_min = z_range[0] + j * delta_z
            z_max = z_min + delta_z

            # Get the list of angle ranges for this square
            square_angle_ranges = angle_ranges[square_index]
            square_index += 1

            # Skip this square if the angle range list is empty
            if not square_angle_ranges:
                continue  # No angles specified, skip this square

            # Create masks for points within this square after rotation
            within_r = (R >= x_min) & (R <= x_max)
            within_z = (Z >= z_min) & (Z <= z_max)

            # Initialize the angular mask for this square
            angular_mask = np.zeros(X.shape, dtype=bool)

            # Process each angle range for this square
            for theta_min, theta_max in square_angle_ranges:
                # Adjust angle ranges to handle wrapping around -pi and pi
                theta_min_adj = theta_min
                theta_max_adj = theta_max
                if theta_min < -np.pi:
                    theta_min_adj += 2 * np.pi
                if theta_max > np.pi:
                    theta_max_adj -= 2 * np.pi

                # Create mask for this angular range
                if theta_min_adj <= theta_max_adj:
                    within_theta = (Theta >= theta_min_adj) & (Theta <= theta_max_adj)
                else:
                    # Angle range crosses the -pi to pi boundary
                    within_theta = (Theta >= theta_min_adj) | (Theta <= theta_max_adj)

                # Update the angular mask
                angular_mask |= within_theta

            # Combine all conditions
            total_mask = within_r & within_z & angular_mask
            binary_matrix |= total_mask

    # Generate the edges of the grid for plotting
    x_edges = np.linspace(-x_range[1], x_range[1], grid_size[0] + 1)
    y_edges = np.linspace(-x_range[1], x_range[1], grid_size[1] + 1)
    z_edges = np.linspace(z_range[0], z_range[1], grid_size[2] + 1)

    return binary_matrix, x_edges, y_edges, z_edges
"""


def generate_binary_matrix(n_x, n_z, x_range, z_range, grid_size, angle_ranges):
    """Generate a binary matrix representing the object based on angle ranges per grid square."""
    # Generate x and z values for the grid in x-z plane
    x_vals_grid = np.linspace(x_range[0], x_range[1], n_x + 1)
    z_vals_grid = np.linspace(z_range[0], z_range[1], n_z + 1)

    # Calculate the increments
    delta_x = (x_range[1] - x_range[0]) / n_x
    delta_z = (z_range[1] - z_range[0]) / n_z

    # Create the 3D grid (voxel centers)
    x_vals = np.linspace(-x_range[1], x_range[1], grid_size[0])
    y_vals = np.linspace(-x_range[1], x_range[1], grid_size[1])
    z_vals = np.linspace(z_range[0], z_range[1], grid_size[2])
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    # Compute radial distance and angular coordinate from Z-axis
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)  # Theta ranges from -pi to pi

    # Initialize the binary matrix
    binary_matrix = np.zeros(X.shape, dtype=bool)

    # Check that angle_ranges has the correct length
    if len(angle_ranges) != n_x * n_z:
        raise ValueError(f"angle_ranges should have length {n_x * n_z}, but has length {len(angle_ranges)}")

    # Adjusted loop order to match generate_square_grid
    square_index = 0
    for i in range(n_x):
        for j in range(n_z):
            # Define the boundaries of the square in x-z plane
            x_min = x_vals_grid[i]
            x_max = x_vals_grid[i + 1]
            z_min = z_vals_grid[j]
            z_max = z_vals_grid[j + 1]

            # Get the list of angle ranges for this square
            square_angle_ranges = angle_ranges[square_index]
            square_index += 1

            # Skip this square if the angle range list is empty
            if not square_angle_ranges:
                continue  # No angles specified, skip this square

            # Create masks for points within this square after rotation
            within_r = (R >= x_min) & (R <= x_max)
            within_z = (Z >= z_min) & (Z <= z_max)

            # Initialize the angular mask for this square
            angular_mask = np.zeros(X.shape, dtype=bool)

            # Process each angle range for this square
            for theta_min, theta_max in square_angle_ranges:
                # Adjust angle ranges to handle wrapping around -pi and pi
                theta_min_adj = theta_min
                theta_max_adj = theta_max
                if theta_min < -np.pi:
                    theta_min_adj += 2 * np.pi
                if theta_max > np.pi:
                    theta_max_adj -= 2 * np.pi

                # Create mask for this angular range
                if theta_min_adj <= theta_max_adj:
                    within_theta = (Theta >= theta_min_adj) & (Theta <= theta_max_adj)
                else:
                    # Angle range crosses the -pi to pi boundary
                    within_theta = (Theta >= theta_min_adj) | (Theta <= theta_max_adj)

                # Update the angular mask
                angular_mask |= within_theta

            # Combine all conditions
            total_mask = within_r & within_z & angular_mask
            binary_matrix |= total_mask

    # Generate the edges of the grid for plotting
    x_edges = np.linspace(-x_range[1], x_range[1], grid_size[0] + 1)
    y_edges = np.linspace(-x_range[1], x_range[1], grid_size[1] + 1)
    z_edges = np.linspace(z_range[0], z_range[1], grid_size[2] + 1)

    return binary_matrix, x_edges, y_edges, z_edges
