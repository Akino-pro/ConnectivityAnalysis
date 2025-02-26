import ast

import numpy as np
from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from helper_functions import compute_length_of_ranges, plot_voronoi_regions_on_sphere, fibonacci_sphere_angles, \
    union_ranges
from spatial3R_ftw_draw import generate_square_grid, draw_rotated_grid
from test_the_end import plot_bar_graph_transposed_same_color
def map_values_to_indices(size, index_list, values):
    """
    Creates a list of length `size`, where values from `values` are placed at
    specified indices in `index_list`, and other indices contain `[]`.

    Parameters:
        size (int): The total length of the output list.
        index_list (list of int or tuples): List of indices or (start, end) ranges.
        values (list): The list containing values to be assigned.

    Returns:
        list: The resulting list with values in specified indices and empty lists elsewhere.
    """
    output = [[] for _ in range(size)]  # Initialize with empty lists

    value_index = 0  # Track values list index

    for index in index_list:
        if isinstance(index, tuple):  # If it's a range (start, end)
            start, end = index
            for i in range(start, end + 1):
                if value_index < len(values):
                    output[i] = values[value_index]
                    value_index += 1
                else:
                    break  # Stop if we run out of values
        else:  # If it's a single index
            if value_index < len(values):
                output[index] = values[value_index]
                value_index += 1

    return output

orientation_samples = 64  # 64
theta_phi_list = fibonacci_sphere_angles(orientation_samples)
with open("my_list.txt", "r") as file:
    content = file.read()
    all_data = ast.literal_eval(content)
index_list_to_color = []
angle_ranges = []
for single_data in all_data:
    index_list_to_color.append(single_data[1])
    beta_range_to_plot = single_data[2]
    alpha_range_to_plot = single_data[3]
    color_list_ori, sm_ori = compute_length_of_ranges(alpha_range_to_plot)
    current_beta = []
    for item in beta_range_to_plot:
        current_beta.extend(item)
        current_beta = union_ranges(current_beta)
    angle_ranges.append(current_beta)
    # plot orientation plot with only fault tolerant orientations with color =alpha range length
    """
    plot_voronoi_regions_on_sphere(theta_phi_list,
                                   beta_range_to_plot,
                                   color_list_ori,
                                   sm_ori,
                                   samples_per_arc=50,
                                   )

    plot_bar_graph_transposed_same_color(theta_phi_list, alpha_range_to_plot)
    """
"""
positional ftw plot
"""
max_length = 5.286553237016408
n_z = int(np.sqrt(2 * 288))
n_x = int(n_z / 2)  # Number of grid divisions along z-axis
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)  #
grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
arc_color = 'blue'
# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
size=288
value_ranges =[(9,13),(32,37),(55,61),(80,84),(104,107),130]
values=angle_ranges
new_list = map_values_to_indices(size, value_ranges, values)
# Set plot range
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
# Draw arcs for each square grid by rotating the entire grid square、
for i, square in enumerate(grid_squares):
    for beta_range in new_list[i]:
        draw_rotated_grid(ax, square, beta_range, arc_color)

# Set plot labels and show the plot
ax.view_init(elev=30, azim=135)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


