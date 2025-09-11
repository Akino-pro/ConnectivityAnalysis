import ast

import numpy as np
from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

from Three_dimension_connectivity_measure import connectivity_analysis
from helper_functions import compute_length_of_ranges, plot_voronoi_regions_on_sphere, fibonacci_sphere_angles, \
    union_ranges, normalize_and_map_colors, computing_6d_volume, get_extruded_wedges, wedge_faces_to_binary_volume, \
    color_by_reliability, plot_alpha_beta_ranges
from spatial3R_ftw_draw import generate_square_grid, draw_rotated_grid, generate_2D_square_grid, generate_grid_centers, \
    generate_binary_matrix
import matplotlib.pyplot as plt
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)


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
max_length = 4.461932111892875
with open("my_list.txt", "r") as file:
    content = file.read()
    all_data = ast.literal_eval(content)
N=len(all_data)
n_z = int(np.sqrt(2 * N))
n_x = int(n_z / 2)  # Number of grid divisions along z-axis
orientation_samples=len(all_data[0][2])
theta_phi_list = fibonacci_sphere_angles(orientation_samples)
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)  #
twod_squares = generate_2D_square_grid(n_x, n_z, x_range, z_range)
grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range) # 64
zeros_list = [0] * N
Sx=(max_length*2*max_length)/N
index_list_to_color = []
index_list_to_color2=[]
angle_ranges = []
V=0
CV=0
sum_connectivity=[]
average_reliability=[]
for single_data in tqdm(all_data, desc="Processing Items"):
    index_list_to_color2.append(single_data[1])
    beta_range_to_plot = single_data[2]
    alpha_range_to_plot = single_data[3]
    average_reliability.append(np.average(single_data[4]))


    vx=computing_6d_volume(alpha_range_to_plot, beta_range_to_plot,grid_centers[single_data[1]], orientation_samples, Sx,single_data[4])
    V+=vx
    zeros_list[single_data[1]]=sum(1 for sublist in alpha_range_to_plot if sublist)/orientation_samples
    color_list_ori, sm_ori = compute_length_of_ranges(alpha_range_to_plot)
    current_beta = []
    ft_tf=False
    alpha_ft_ranges=[]
    for beta_index,item in enumerate(beta_range_to_plot):
        if single_data[4][beta_index]==1:
            ft_tf=True
            alpha_ft_ranges.append(alpha_range_to_plot[beta_index])
            current_beta.extend(item)
        else:alpha_ft_ranges.append([])
    if all(len(sublist) == 0 for sublist in alpha_range_to_plot) or all(len(sublist) == 0 for sublist in alpha_ft_ranges): sum_connectivity.append(0)
    else:
            all_wedge_faces = get_extruded_wedges(
                theta_phi_list,
                alpha_ft_ranges,
                extrude_radius=2 * np.pi,
            )
            #binary_volume = wedge_faces_to_binary_volume(all_wedge_faces, NX=50, NY=50, NZ=50)
            #shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_volume,
            #                                                                                 2, 0.5)
            #sum_connectivity.append(connected_connectivity)
    if ft_tf:
        current_beta = union_ranges(current_beta)
        angle_ranges.append(current_beta)
        index_list_to_color.append(single_data[1])
    # plot orientation plot with only fault tolerant orientations with color =alpha range length
print(f'The 6D volume of FT workspace is {V}.')
#colors,sm=normalize_and_map_colors(zeros_list, cmap_name='rainbow')
colors,sm = color_by_reliability(average_reliability)



fig, ax = plt.subplots()
ax.set_xlim([0, max_length])
ax.set_ylim([-max_length, max_length])
ax.set_aspect(1)
for i, square in enumerate(twod_squares):
    color = colors[i]
    color = 'white'
    alpha_level = 1.0
    if i not in index_list_to_color2:
        color='white'
    polygon = patches.Polygon(square, facecolor=color, edgecolor='k', alpha=alpha_level, linewidth=1)
    ax.add_patch(polygon)

# Set plot labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Z')
frame_points = [
    (x_range[0], z_range[0]), (x_range[1], z_range[0]),
    (x_range[1], z_range[1]), (x_range[0], z_range[1]),
    (x_range[0], z_range[0])  # Closing the loop
]

# Draw frame
frame_x, frame_z = zip(*frame_points)
ax.plot(frame_x, frame_z, color='k', linewidth=2)
#tick_values = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
#tick_labels = [f'{int(t * 100)}%' for t in tick_values]
#cbar = plt.colorbar(sm, ticks=tick_values,ax=ax)
#cbar.ax.set_yticklabels(tick_labels)
cbar = plt.colorbar(sm, ax=ax)
plt.draw()
print("Press 'q' to continue...")
while True:
    key = plt.waitforbuttonpress()
    if key:
        break

plt.close(fig)




index_of_chioce=[9,34,83]
#index_of_chioce=[8,16,18] #origin al
#index_of_chioce=[8,15,17] #shifted 30
#index_of_chioce=[3,10,12] #shrinked 30
#index_of_chioce=[8,15,17] #expanded 30
#index_of_chioce=[8,15,17] #shifted 60
#index_of_chioce=[8,15,17] #shifted 90
for ind in index_of_chioce:
    #color_list_ori,sm_ori=compute_length_of_ranges(all_data[ind][3])
    #color_list_ori, sm_ori = color_by_reliability(all_data[ind][4])
    color_list_ori, sm_ori = normalize_and_map_colors(all_data[ind][5])
    plot_voronoi_regions_on_sphere(theta_phi_list,
                                   color_list_ori,
                                   sm_ori
                                   )
    plot_alpha_beta_ranges(theta_phi_list, all_data[ind][3],all_data[ind][2])
    #plot_alpha_ranges(theta_phi_list, all_data[ind][3])
    #plot_beta_ranges(theta_phi_list,  all_data[ind][2])





#positional ftw plot

max_length = 4.461932111892875
n_z = int(np.sqrt(2 * N))
n_x = int(n_z / 2)  # Number of grid divisions along z-axis
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)  #
grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)

# Plot setup
size=N
new_list = [[] for _ in range(N)]
counter=0
for index in index_list_to_color:
    new_list[index]=angle_ranges[counter]
    counter+=1
"""
# Set plot range
arc_color = 'blue'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
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
"""
grid_size = (64, 64, 64)
binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
        n_x, n_z, x_range, z_range, grid_size, new_list
    )
shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_matrix,1,0.5)

C=0.5*(connected_connectivity+np.average(sum_connectivity))
print(f'The 6D connectivity of FT workspace is {C}.')
CV=C*V
print(f'The 6D connectivity*volume of FT workspace is {CV}.')
