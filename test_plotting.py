import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def generate_binary_matrix(n_x, n_z, x_range, z_range, grid_size, angle_ranges):
    """
    Generate a binary 3D matrix representing the rotated grid object with individual angle ranges per grid square,
    where each grid square can have multiple angle ranges or be empty.

    Parameters:
    - n_x, n_z: Number of divisions along x and z axes in the grid.
    - x_range, z_range: Tuples specifying the range of x and z values.
    - grid_size: Tuple specifying the number of points along each axis in the 3D grid.
    - angle_ranges: List where each element corresponds to a grid square and contains:
        - An empty list [] if no angles are specified for that grid square.
        - A list of one or more angle ranges [(theta_min, theta_max), ...] for that grid square.

    Returns:
    - binary_matrix: A 3D NumPy array with True inside the object and False outside.
    - x_edges, y_edges, z_edges: 1D arrays of the x, y, z coordinates at the edges of the grid.
    """
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
# Generate the binary matrix with the specified angle ranges per square
binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
    n_x, n_z, x_range, z_range, grid_size, angle_ranges
)

# Create coordinate arrays for the voxel edges
x, y, z = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

# Plot the binary matrix using the coordinate arrays
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the binary matrix and the coordinate arrays to plot the voxels
ax.voxels(x, y, z, binary_matrix, facecolors='blue', edgecolor='k', alpha=0.7)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([x_edges[0], x_edges[-1]])
ax.set_ylim([y_edges[0], y_edges[-1]])
ax.set_zlim([z_edges[0], z_edges[-1]])
plt.title('Binary Matrix with Individual Angle Ranges per Grid Square')
plt.show()
"""


# Parameters
# Parameters
N = 50
n_z = int(np.sqrt(2 * N))
n_x = int(n_z / 2)
x_range = (0, 3)
z_range = (-3, 3)
grid_size = (32, 32, 32)  # Adjust for desired resolution

# Create angle ranges for each grid square
# Initialize angle_ranges as a list with desired angle ranges for each square
angle_ranges = []

for idx in range(n_x * n_z):
    # Example configurations
    if idx % 5 == 0:
        # For every 5th square, no angles (empty), so nothing is drawn
        angle_ranges.append([])
    elif idx % 2 == 0:
        # For even indices, multiple angle ranges
        angle_ranges.append([
            (-np.pi, -np.pi / 2),
            (np.pi / 2, np.pi)
        ])
    else:
        # For odd indices, a single angle range
        angle_ranges.append([
            (-np.pi / 4, np.pi / 4)
        ])

# Generate the binary matrix with the specified angle ranges per square
binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
    n_x, n_z, x_range, z_range, grid_size, angle_ranges
)

# Use Marching Cubes to extract the surface
# Pad the volume to ensure proper surface extraction at the edges
padded_volume = np.pad(binary_matrix.astype(np.uint8), pad_width=1, mode='constant', constant_values=0)
verts, faces, normals, values = measure.marching_cubes(padded_volume, level=0.5)

# Adjust the coordinates to match the actual grid
# Compute the spacing between grid points
spacing = (
    (x_edges[-1] - x_edges[0]) / (grid_size[0]),
    (y_edges[-1] - y_edges[0]) / (grid_size[1]),
    (z_edges[-1] - z_edges[0]) / (grid_size[2])
)

# Shift the vertices according to the grid spacing and edges
verts[:, 0] = x_edges[0] - spacing[0] + verts[:, 0] * spacing[0]
verts[:, 1] = y_edges[0] - spacing[1] + verts[:, 1] * spacing[1]
verts[:, 2] = z_edges[0] - spacing[2] + verts[:, 2] * spacing[2]

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a triangular mesh plot
ax.plot_trisurf(
    verts[:, 0], verts[:, 1], verts[:, 2],
    triangles=faces, linewidth=0.2, antialiased=True, color='blue', alpha=0.7
)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([x_edges[0], x_edges[-1]])
ax.set_ylim([y_edges[0], y_edges[-1]])
ax.set_zlim([z_edges[0], z_edges[-1]])
plt.title('Smooth 3D Object with Variable Angle Ranges per Grid Square')
plt.show()