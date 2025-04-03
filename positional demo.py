import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spatial3R_ftw_draw import rotate_around_z


def draw_rotated_grid(ax, square, angle_range, color,alpha):
    intersected_lower = angle_range[0]
    intersected_upper = angle_range[1]
    num=int((intersected_upper-intersected_lower)/(np.pi/180)*3)
    print(num)
    # if square[0][2] >= 0:
    #    if angle_range[0] >= 0 or angle_range[1] <= -np.pi / 2: return
    #    intersected_lower = np.max([angle_range[0], -np.pi / 2])
    #    intersected_upper = np.min([angle_range[1], 0])

    angles = np.linspace(intersected_lower, intersected_upper, num)

    # Loop through each angle to rotate the square around the Z-axis
    for angle in angles:
        rotated_vertices = rotate_around_z(np.array(square), angle)

        # Create the rotated face (polygon) for the square
        face = [rotated_vertices[j] for j in range(4)]

        # Add the polygon to the plot with a single color
        ax.add_collection3d(Poly3DCollection([face], color=color,alpha=alpha))

def generate_square_grid(n_x, n_z, x_range, z_range):
    """Generate square patches covering the grid."""
    x_step = (x_range[1] - x_range[0]) / n_x
    z_step = (z_range[1] - z_range[0]) / n_z

    squares = []
    for j in range(n_z):
        for i in range(n_x):
            x_start = x_range[0] + i * x_step
            z_start = z_range[0] + j * z_step

            square = [
                [x_start, 0, z_start],
                [x_start + x_step, 0, z_start],
                [x_start + x_step, 0, z_start + z_step],
                [x_start, 0, z_start + z_step]
            ]
            squares.append(square)
    return squares


N = 200
n_z = int(np.sqrt(2 * N))
n_x = max(1, int(n_z / 2))  # Ensure n_x is at least 1
max_length = 1

x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)  # Range for z-axis

def generate_grid_centers(n_x, n_z, num_points, x_range, z_range):
    """Generate the center points of each grid in the x-z plane with correct order."""
    if n_x * n_z != num_points:
        raise ValueError(f"Number of points ({num_points}) does not match grid layout ({n_x} * {n_z})")

    x_step = (x_range[1] - x_range[0]) / n_x
    z_step = (z_range[1] - z_range[0]) / n_z

    centers = []
    for j in range(n_z):
        for i in range(n_x):
            x_center = x_range[0] + (i + 0.5) * x_step
            z_center = z_range[0] + (j + 0.5) * z_step
            centers.append([x_center, 0, z_center])  # y = 0 as we're in the x-z plane

    return np.array(centers)

# Generate grid data
grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)

# Plot setup
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set plot range
ax.set_xlim([-max_length, max_length])
ax.set_ylim([-max_length, max_length])
ax.set_zlim([-max_length, max_length])
ax.set_box_aspect([1, 1, 1])

# Draw grid squares
for idx, square in enumerate(grid_squares):
    color = 'green'
    alpha_level = 0.1

    # Modify the 100th square
    if idx == 105:  # Index 99 corresponds to the 100th element (0-based index)
        edge_color = 'green'
        line_width = 5
    else:
        edge_color = 'k'
        line_width = 0.1

    square_poly = Poly3DCollection([square], facecolor=color, edgecolor=edge_color, linewidth=line_width,
                                   alpha=alpha_level)
    ax.add_collection3d(square_poly)

# Plot grid centers as black points
ax.scatter(grid_centers[:, 0], grid_centers[:, 1], grid_centers[:, 2], color='black', s=1)
angle_ranges=[[-3*np.pi/4,-np.pi/2],[-np.pi,-3*np.pi/4],[np.pi/2,np.pi],[-np.pi/2,np.pi/2]]
counter=0
alpha_list=[0.7,1,0.4,0.1]
color_list=  [
    "#388E3C" ,"#1B5E20","#4CAF50","#A5D6A7"
]
for beta_range in angle_ranges:
    draw_rotated_grid(ax, grid_squares[105], beta_range, color_list[counter],alpha=1)
    counter+=1


# Remove ticks and add labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
ax.set_zlabel('z',fontsize=18)

# Show legend and plot
ax.legend()
plt.show()
