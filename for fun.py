import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spatial3R_ftw_draw import generate_square_grid_centers, generate_square_grid, draw_rotated_grid, \
    draw_rotated_grid_by_range_only, draw_rotated_grid_by_range_only_transparent, rotate_around_z

N = 200
n_z = int(np.sqrt(2 * N))
n_x = int(n_z / 2)  # Number of grid divisions along z-axis
max_length = 1
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)  #
grid_squares_centers = generate_square_grid_centers(n_x, n_z, x_range, z_range)
grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot squares with green color and semi-transparency (alpha=0.5)
counter = 0
for square in grid_squares:
    alpha = 0.1
    linewidth = 0.2
    edgecolor = 'k'
    if counter == 89: linewidth = 5;edgecolor = 'green'
    poly = Poly3DCollection([square], facecolors='green', alpha=alpha, linewidths=linewidth, edgecolors=edgecolor)
    counter += 1
    ax.add_collection3d(poly)
centers = np.array(grid_squares_centers)
# Plot centers as black points
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='black', s=1)

# Set bounds
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z, rotation axis of joint')
target_grid = grid_squares[89]
sample = [[0, np.pi / 2], [np.pi / 2, np.pi], [-np.pi, -3 * np.pi / 4], [-3 * np.pi / 4, -np.pi / 2], [-np.pi / 2, 0]]
alpha_list = [0.3, 0.7, 1, 0.5, 0.3]
color_list = ["#C8E6C9", "#81C784", "#388E3C", "#4CAF50", "#C8E6C9"]
for i in range(5):
    draw_rotated_grid_by_range_only(ax, target_grid, sample[i], color_list[i])


plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
draw_rotated_grid_by_range_only_transparent(ax, grid_squares[89], [-np.pi, -3 * np.pi / 4], "#388E3C", grid_squares_centers[89])
selected_center=grid_squares_centers[89]
example_angle_range=[-np.pi, -3 * np.pi / 4]
center_lower = rotate_around_z(np.array([selected_center]), example_angle_range[0])[0].flatten()
center_upper = rotate_around_z(np.array([selected_center]), example_angle_range[1])[0].flatten()

# Plot centers of faces explicitly
ax.scatter(*center_lower, color='k', s=10)
ax.scatter(*center_upper, color='k', s=10)

# Set bounds
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.grid(False)                          # Remove the grid
ax.set_axis_off()                       # Remove axis lines, ticks, and labels
ax.xaxis.pane.set_edgecolor('w')        # Remove x-axis pane edges
ax.yaxis.pane.set_edgecolor('w')        # Remove y-axis pane edges
ax.zaxis.pane.set_edgecolor('w')        # Remove z-axis pane edges

ax.xaxis.pane.set_facecolor((1,1,1,0))  # Set transparent background
ax.yaxis.pane.set_facecolor((1,1,1,0))
ax.zaxis.pane.set_facecolor((1,1,1,0))

ax.xaxis._axinfo['grid'].update(color = (1,1,1,0))  # Remove internal grids completely
ax.yaxis._axinfo['grid'].update(color = (1,1,1,0))
ax.zaxis._axinfo['grid'].update(color = (1,1,1,0))


plt.show()
