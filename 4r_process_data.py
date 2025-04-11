import ast

import numpy as np
from matplotlib import pyplot as plt

from Three_dimension_connectivity_measure import connectivity_analysis
from helper_functions import normalize_and_map_colors, sorted_indices, update_or_add_square_2d
from spatial3R_ftw_draw import generate_grid_centers, generate_binary_matrix, generate_2D_square_grid

with open("my_list.txt", "r") as file:
    content = file.read()
    all_data = ast.literal_eval(content)

N=5000
max_length =3.1989682240512938
n_z = int(np.sqrt(2 * N))
n_x = int(n_z / 2)  # Number of grid divisions along z-axis
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)
Sx=(max_length*2*max_length)/N
ftw_beta_ranges=all_data[-1]
print(len(ftw_beta_ranges))
vx=0
grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
for index,non_overlap_betas in enumerate(ftw_beta_ranges):
    for beta_range in non_overlap_betas:
        vx += grid_centers[index][0] * Sx * (beta_range[1] - beta_range[0])
print(f'the failure tolerant workspace volume is {vx}.')

grid_size = (64, 64, 64)
binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
        n_x, n_z, x_range, z_range, grid_size, ftw_beta_ranges
    )
shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_matrix,1,0.5)
print(f'the failure tolerant workspace connectivity is {connected_connectivity}.')

r1 = 0.5
r2 = 0.6
r3 = 0.7
r4 = 0.8

#1,2,3,4,12,13,14,23,24,34,123,124,134,234,1234

def reliability_computation(r1, r2, r3, r4):
    reliability_list = [r2 * r3 * r4, r1 * r3 * r4, r1 * r2 * r4, r1 * r2 * r3,
                        r1 * r3 * r4 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r2 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r1 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r3 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 - r1 * r2 * r3 * r4,
                        r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
                        r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4]
    # reliability_list = [r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r3 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r2 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 - 2 * r1 * r2 * r3 * r4,
    #                    r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4]
    conditional_reliability_list = []
    for p in reliability_list:
        conditional_reliability_list.append(
            (p - r1 * r2 * r3 * r4) /
            (r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 4 * r1 * r2 * r3 * r4))
    return conditional_reliability_list


cr_list = reliability_computation(r1, r2, r3, r4)

color_list, sm = normalize_and_map_colors(cr_list)
fig, ax2 = plt.subplots()
ax2.set_xlim([0, max_length])
ax2.set_ylim([-max_length, max_length])
ax2.set_aspect(1)
cbar = plt.colorbar(sm, ax=ax2, label='Reliability Spectrum')

plt.ion()
indices = sorted_indices(cr_list)
index_dict = {}
for you in indices:
    ftw_points_count = 0

    """
    arc_color = color_list[you]

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-3, 3])  # todo: optimized
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    for i, square in tqdm(enumerate(grid_squares), desc="Processing Items"):
        for beta_range in all_reliable_beta_ranges[you][i]:
            # draw_wedge(ax, square, beta_range, arc_color)
            draw_rotated_grid(ax, square, beta_range, arc_color)


    ax.view_init(elev=30, azim=135)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    print("Press 'q' to continue...")
    while True:
        key = plt.waitforbuttonpress()
        if key:  # Any key will work, but we can restrict it if needed
            break

    plt.close(fig)
    """
    """
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([0, max_length])
    ax.set_ylim([-max_length, max_length])
    ax.set_zlim([-max_length, max_length])
    ax.set_box_aspect([1, 1, 2])

    # Draw squares only if the angle_ranges[i] is non-empty
    for i, square in enumerate(grid_squares):
        color = 'k'
        alpha_level = 0
        if all_reliable_beta_ranges[you][i]:  # Check if the list is non-empty
            color = color_list[you]
            alpha_level = 1.0
            ftw_points_count += 1
            # Plot the square grid directly
        square_poly = Poly3DCollection([square], facecolor=color, edgecolor='k', alpha=alpha_level)
        ax.add_collection3d(square_poly)


    # Set plot labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.draw()
    print("Press 'q' to continue...")
    while True:
        key = plt.waitforbuttonpress()
        if key:
            break

    plt.close(fig)

    print(f'we have {ftw_points_count} grids over {grid_sample_num} fault tolerant')
    # also plot a 2D view of it
    """
    twod_squares = generate_2D_square_grid(n_x, n_z, x_range, z_range)
    # fig, ax = plt.subplots()
    # ax.set_xlim([0, max_length])
    # ax.set_ylim([-max_length, max_length])
    # ax.set_aspect(1)
    for i, square in enumerate(twod_squares):
        color = 'w'
        alpha_level = 1.0
        if all_data[you][i]:
            color = color_list[you]
            ftw_points_count += 1

        # polygon = patches.Polygon(square, facecolor=color, edgecolor='k', alpha=alpha_level, linewidth=0.5)
        # ax.add_patch(polygon)
        update_or_add_square_2d(ax2, square, color, alpha_level, i, index_dict=index_dict)
    """
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

    plt.draw()
    print("Press 'q' to continue...")
    while True:
        key = plt.waitforbuttonpress()
        if key:
            break

    plt.close(fig)
    """

ax2.set_xlabel('X')
ax2.set_ylabel('Z')
frame_points = [
    (x_range[0], z_range[0]), (x_range[1], z_range[0]),
    (x_range[1], z_range[1]), (x_range[0], z_range[1]),
    (x_range[0], z_range[0])  # Closing the loop
]

# Draw frame
frame_x, frame_z = zip(*frame_points)
ax2.plot(frame_x, frame_z, color='k', linewidth=2)
plt.ioff()
plt.show()