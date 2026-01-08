import time

import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tqdm import tqdm
from scipy.linalg import null_space
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3

from helper_functions import normalize_and_map_colors, update_or_add_square, sorted_indices, union_ranges, \
    update_or_add_square_2d, update_beta_bar_multicolor, key3, set_sparse_xyz_labels, draw_cube, set_axes_equal, \
    draw_sphere
from spatial3R_ftw_draw import generate_grid_centers, generate_square_grid, draw_rotated_grid, generate_2D_square_grid
from Three_dimension_connectivity_measure import connectivity_analysis
from spatial3R_ftw_draw import generate_binary_matrix

kernel_size = 1
Lambda = 0.5
step_size = 0.01
terminate_threshold = 9.0 / 5.0 * step_size
# terminate_threshold = step_size * 0.5
ssm_finding_num = 20
max_ssm = 4
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
            p /
            (r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4))
    return conditional_reliability_list


cr_list = reliability_computation(r1, r2, r3, r4)
print(cr_list)
print(sorted_indices(cr_list))


# print(cr_list)


def convert_to_C_dot_A(CA):
    """
    Converts a list of ranges CA to C_dot_A by handling wrap-around cases around [-π, π].

    Parameters:
        CA (list of tuples): List of ranges, each as (lower, upper).

    Returns:
        list of lists of tuples: C_dot_A with each original range and any additional ranges to handle wrap-around.
    """
    C_dot_A = []

    for lower, upper in CA:
        # Initialize the sublist for the current range
        sublist = [(lower, upper)]

        # Check if the lower limit is below -π
        if lower < -np.pi:
            # Add [lower + 2π, upper + 2π] to handle wrap-around
            sublist.append((lower + 2 * np.pi, upper + 2 * np.pi))

        # Check if the upper limit is above π
        if upper > np.pi:
            # Add [lower - 2π, upper - 2π] to handle wrap-around
            sublist.append((lower - 2 * np.pi, upper - 2 * np.pi))

        # Append the sublist to C_dot_A
        C_dot_A.append(sublist)

    return C_dot_A


def convert_plot_to_voxel_matrix(ax, grid_size=100, plot_range=(-4, 4)):
    """
    Convert a Matplotlib 3D plot to a binary 3D matrix (voxel grid) by checking if each voxel
    overlaps with the plot data in ax.

    Parameters:
    ax : Matplotlib 3D axis object that contains the 3D plot data.
    grid_size : The size of the 3D grid (number of voxels along each axis), default is 100x100x100.
    plot_range : Tuple representing the range of the plot along each axis.

    Returns:
    voxel_matrix : A binary 3D matrix (voxel grid) representing the 3D plot in ax.
    """
    # Initialize an empty voxel matrix with the given size (0 for empty, 1 for filled)
    voxel_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Define the resolution for each axis based on plot range and grid size
    plot_min, plot_max = plot_range
    voxel_resolution = (plot_max - plot_min) / grid_size

    # Generate the center coordinates for each voxel in the grid
    x_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2
    y_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2
    z_coords = np.linspace(plot_min, plot_max, grid_size) + voxel_resolution / 2

    # Check each voxel position to see if it intersects with plot elements in ax
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                # Check if the voxel at (x, y, z) overlaps with any of the plot elements
                # ax contains function can check if a point lies within plotted data (example only)
                # Note: for precise checks, you'd need to develop custom intersection logic
                point_in_plot = any(collection.contains_point([x, y, z]) for collection in ax.collections)

                # Update the voxel matrix
                if point_in_plot:
                    voxel_matrix[i, j, k] = 1

    return voxel_matrix


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


def stepwise_ssm(theta, n_j, x_target, previous_n_j, robot):
    x = robot.fkine(theta.flatten())
    x = np.array(x)[:3, 3].T.reshape((3, 1))
    delta_x_step = x_target - x
    J_step = robot.jacob0(theta.flatten())[:3, :]
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + dq * step_size
    # x_next = robot.fkine(theta_next.flatten())
    # print(x_next - x_target)
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    new_J = robot.jacob0(theta_next.flatten())[:3, :]
    new_n_j = null_space(new_J)
    m, n = np.shape(new_n_j)
    if n == 2:
        print('special case')
        n_j_1 = new_n_j[:, 0]
        n_j_2 = new_n_j[:, 1]
        if np.dot(n_j_1, previous_n_j) > np.dot(n_j_2, previous_n_j):
            new_n_j = n_j_1
        else:
            new_n_j = n_j_2
    a = np.array(new_n_j).reshape(-1)  # Converts 4x1 to 1D array
    a_prime = np.array(-new_n_j).reshape(-1)
    b = np.array(previous_n_j).reshape(-1)
    dot_product = np.dot(a, b)
    dot_product_prime = np.dot(a_prime, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_a_prime = np.linalg.norm(a_prime)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta_prime = dot_product_prime / (norm_a_prime * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    cos_theta_prime = np.clip(cos_theta_prime, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_rad_prime = np.arccos(cos_theta_prime)
    if np.abs(angle_rad) > np.abs(angle_rad_prime):
        new_n_j = -new_n_j

    return theta_next, new_n_j, n_j


def find_intersection_points(ssm_theta_list, C_dot_A):
    tof = False
    counter = 0
    ip_ranges = []
    isin_list = []
    for theta_index in range(len(ssm_theta_list)):
        theta = ssm_theta_list[theta_index]
        theta_flatten = theta.flatten()
        if all(any(r[0] <= v <= r[1] for r in ranges) for v, ranges in zip(theta_flatten, C_dot_A)):
            isin_list.append(theta_index)
            tof = True
            counter += 1
    ip_index_ranges = []
    if len(isin_list) != 0: ip_index_ranges = find_ranges(isin_list)

    for index_range in ip_index_ranges:
        local_min_theta1 = 100
        local_max_theta1 = -100
        for i in range(index_range[0], index_range[1] + 1):
            if ssm_theta_list[i][0][0] < local_min_theta1:
                local_min_theta1 = ssm_theta_list[i][0][0]
            if ssm_theta_list[i][0][0] > local_max_theta1:
                local_max_theta1 = ssm_theta_list[i][0][0]
        if np.abs(local_min_theta1 + np.pi) <= 1e-2: local_min_theta1 = -np.pi
        if np.abs(local_min_theta1 - np.pi) <= 1e-2: local_min_theta1 = np.pi
        if np.abs(local_max_theta1 + np.pi) <= 1e-2: local_max_theta1 = -np.pi
        if np.abs(local_max_theta1 - np.pi) <= 1e-2: local_max_theta1 = np.pi
        ip_ranges.append([local_min_theta1, local_max_theta1])
        # print([local_min_theta1, local_max_theta1])
    return union_ranges(ip_ranges), tof


def find_ranges(nums):
    # Sort the input list
    nums = sorted(set(nums))
    ranges = []
    start = nums[0]

    for i_fr in range(1, len(nums)):
        # If the current number is not consecutive
        if nums[i_fr] != nums[i_fr - 1] + 1:
            ranges.append([start, nums[i_fr - 1]])
            start = nums[i_fr]

    # Append the last range
    ranges.append([start, nums[-1]])
    return ranges


def extend_ranges(unioned_ranges):
    extended_ranges = []
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in unioned_ranges:
        if tr[0] == -np.pi and tr[1] == np.pi:
            return [[- np.pi, np.pi]]
        elif tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        elif tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        else:
            extended_ranges.append(tr)

    if has_negative_pi_range is not None and has_positive_pi_range is not None:
        extended_ranges.append([has_positive_pi_range - 2 * np.pi, has_negative_pi_range])
        extended_ranges.append([has_positive_pi_range, has_negative_pi_range + 2 * np.pi])
    elif has_negative_pi_range is not None:
        extended_ranges.append([-np.pi, has_negative_pi_range])
    elif has_positive_pi_range is not None:
        extended_ranges.append([has_positive_pi_range, np.pi])
    return extended_ranges


def find_critical_points(ssm_theta_list):
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
                , [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        thetas_cp_range[3] = [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
        previous_theta = ssm_theta_list[0]
        ssm_theta_list.append(previous_theta)
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(4):
                #if i==3: print(theta[i][0])
                if np.abs(theta[i][0] - previous_theta[i][0]) > 3:
                    if theta[i][0] > previous_theta[i][0]:
                        thetas_cp_sum[i].append([-np.pi, thetas_cp_range[i][1]])
                        thetas_cp_range[i] = [theta[i][0], np.pi]
                    else:
                        thetas_cp_sum[i].append([thetas_cp_range[i][0], np.pi])
                        thetas_cp_range[i] = [-np.pi, theta[i][0]]
                elif theta[i][0] < thetas_cp_range[i][0]:
                    thetas_cp_range[i][0] = theta[i][0]
                elif theta[i][0] > thetas_cp_range[i][1]:
                    thetas_cp_range[i][1] = theta[i][0]
            previous_theta = theta
    for j in range(4):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
        #if j == 0: print(thetas_cp_sum[j])
    # print(thetas_cp_sum)
    return thetas_cp_sum

def find_single_intersection(ssm_theta_list):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof

def find_random_ssm(x_target, all_ssm_theta_list, robot, C_dot_A):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(4,))).T.reshape((4, 1))
    Tep = SE3(x_target)
    result = robot.ikine_LM(Tep, mask=[1, 1, 1, 0, 0, 0], q0=q)
    if not result.success: return result.success, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    sol = result.q.T.reshape((4, 1))
    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    theta = sol
    theta_prime = theta.copy()

    J = robot.jacob0(theta.flatten())[:3, :]
    n_j = null_space(J)[:, 0].T.reshape((4, 1))
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    while True:
        num += 1
        if threshold <= terminate_threshold and num >= 4: break
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j, robot)
        threshold = np.linalg.norm(theta - theta_prime)
        # print(threshold)
        if threshold < lowest: lowest = threshold

        if num == 100:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
        n_j = new_n_j
        if num > 5000:
            print('stuck at a too small smm')
            theta_prime = theta
            num = 0
        ssm_theta_list.append(theta)

    """
    points = np.array(ssm_theta_list)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')

        # Set labels
    plt.xlabel('theta1')
    plt.ylabel('theta2')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 1], points[:, 2], c='b', marker='o')

        # Set labels
    plt.xlabel('theta2')
    plt.ylabel('theta3')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 2], points[:, 3], c='b', marker='o')

        # Set labels
    plt.xlabel('theta3')
    plt.ylabel('theta4')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()

    plt.figure()
    plt.scatter(points[:, 3], points[:, 0], c='b', marker='o')

        # Set labels
    plt.xlabel('theta4')
    plt.ylabel('theta1')

        # Set plot limits for better visualization
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.show()
    """


    all_ssm_theta_list.extend(ssm_theta_list)
    # print(f'found a new ssm with {num} points.')
    ssm_found = True

    ip_ranges, tof = find_intersection_points(ssm_theta_list, C_dot_A)
    cp_ranges = find_critical_points(ssm_theta_list)
    single_intersection_tf = find_single_intersection(ssm_theta_list)
    return True, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found,single_intersection_tf


def compute_beta_range(x, y, z, robot, C_dot_A, CA):
    target_x = np.array([x, y, z]).T.reshape((3, 1))
    all_smm_beta_range = []
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    find_count = 0
    ssm_found = 0
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf = find_random_ssm(
            target_x, all_theta, robot, C_dot_A)
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            # print(beta0_lm, beta0_um)
            if beta0_um - beta0_lm >= 2 * np.pi:
                beta0_ranges.append([-np.pi, np.pi])
            elif beta0_lm < -np.pi:
                beta0_ranges.append([beta0_lm + 2 * np.pi, np.pi])
                beta0_ranges.append([-np.pi, beta0_um])
            elif beta0_um > np.pi:
                beta0_ranges.append([-np.pi, beta0_um - 2 * np.pi])
                beta0_ranges.append([beta0_lm, np.pi])
            else:
                beta0_ranges.append([beta0_lm, beta0_um])
        if ssm_found_tf:  ssm_found += 1; find_count = 0
        for cp_range in cp_ranges[0]:
            if cp_range[0] > -np.inf: theta1_ranges.append(cp_range)
        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf: theta2_ranges.append(cp_range)
        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf: theta3_ranges.append(cp_range)
        for cp_range in cp_ranges[3]:
            if cp_range[0] > -np.inf: theta4_ranges.append(cp_range)

    # print(beta0_ranges)
    if len(theta1_ranges) == 0: return []
    if len(theta2_ranges) == 0: return []
    if len(theta3_ranges) == 0: return []
    if len(theta4_ranges) == 0: return []
    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)

    # print(theta1_ranges_union)
    # print(theta2_ranges_union)
    # print(theta3_ranges_union)
    # print(theta4_ranges_union)
    # print('hello')
    # print(extend_ranges(theta1_ranges_union))
    # print(extend_ranges(theta2_ranges_union))
    # print(extend_ranges(theta3_ranges_union))
    # print(extend_ranges(theta4_ranges_union))

    if len(all_theta) != 0:
        points = np.array(all_theta)

        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c='b', marker='o')

        # Set labels
        plt.xlabel('theta1')
        plt.ylabel('theta2')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 1], points[:, 2], c='b', marker='o')

        # Set labels
        plt.xlabel('theta2')
        plt.ylabel('theta3')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 2], points[:, 3], c='b', marker='o')

        # Set labels
        plt.xlabel('theta3')
        plt.ylabel('theta4')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

        plt.figure()
        plt.scatter(points[:, 3], points[:, 0], c='b', marker='o')

        # Set labels
        plt.xlabel('theta4')
        plt.ylabel('theta1')

        # Set plot limits for better visualization
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])

        plt.show()

    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            # print('joint 1 succeed')
            min_beta1 = CA[0][1] - tr[1]
            max_beta1 = CA[0][0] - tr[0]
            if (max_beta1 - min_beta1 >= 2 * np.pi) or (tr[0] == -np.pi and tr[1] == np.pi):
                max_beta1 = np.pi
                min_beta1 = -np.pi

    ion2 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta2_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[1][0] >= tr[0] and CA[1][1] <= tr[1]:
            ion2 = True
            # print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')

    ion3 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta3_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[2][0] >= tr[0] and CA[2][1] <= tr[1]:
            ion3 = True
            # print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')

    ion4 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta4_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[3][0] >= tr[0] and CA[3][1] <= tr[1]:
            ion4 = True
            # print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')

    # for index in range(len(beta0_ranges)):
    #    min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
    #    min_beta_f_ftw = -np.pi
    #   max_beta_f_ftw = np.pi
    #    all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
    if ion1 and ion2 and ion3 and ion4:
        for index in range(len(beta0_ranges)):
            min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
            # print(min_beta0, max_beta0)
            # print(min_beta1, max_beta1)
            # print(min_beta2, max_beta2)
            # print(min_beta3, max_beta3)
            # print(min_beta3, max_beta4)

            min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
            max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
            if min_beta_f_ftw <= max_beta_f_ftw:
                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
            # reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])

    # for rbr_index in range(len(reliable_beta_ranges)):
    #    reliable_beta_ranges[rbr_index] = union_ranges(reliable_beta_ranges[rbr_index])
    # print(all_smm_beta_range)
    return union_ranges(all_smm_beta_range)


def compute_reliable_beta_range(x, y, z, robot, C_dot_A, CA, all_reliable_beta_ranges):
    # 15 cases
    reliable_beta_ranges = [[] for _ in range(16)]
    target_x = np.array([x, y, z]).T.reshape((3, 1))
    F_list=[False] * 16
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    find_count = 0
    ssm_found = 0
    single_intersection_tf_all = False
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf,single_intersection_tf = find_random_ssm(
            target_x, all_theta, robot, C_dot_A)
        single_intersection_tf_all = single_intersection_tf_all or single_intersection_tf
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            # print(beta0_lm, beta0_um)
            if beta0_um - beta0_lm >= 2 * np.pi:
                beta0_ranges.append([-np.pi, np.pi])
            elif beta0_lm < -np.pi:
                beta0_ranges.append([beta0_lm + 2 * np.pi, np.pi])
                beta0_ranges.append([-np.pi, beta0_um])
            elif beta0_um > np.pi:
                beta0_ranges.append([-np.pi, beta0_um - 2 * np.pi])
                beta0_ranges.append([beta0_lm, np.pi])
            else:
                beta0_ranges.append([beta0_lm, beta0_um])
        if ssm_found_tf:  ssm_found += 1; find_count = 0
        for cp_range in cp_ranges[0]:
            if cp_range[0] > -np.inf: theta1_ranges.append(cp_range)
        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf: theta2_ranges.append(cp_range)
        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf: theta3_ranges.append(cp_range)
        for cp_range in cp_ranges[3]:
            if cp_range[0] > -np.inf: theta4_ranges.append(cp_range)
    # print(beta0_ranges)
    # if len(theta1_ranges) == 0: return reliable_beta_ranges
    # if len(theta2_ranges) == 0: return reliable_beta_ranges
    # if len(theta3_ranges) == 0: return reliable_beta_ranges
    # if len(theta4_ranges) == 0: return reliable_beta_ranges
    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)
    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)

    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            # print('joint 1 succeed')
            min_beta1 = CA[0][1] - tr[1]
            max_beta1 = CA[0][0] - tr[0]
            if (max_beta1 - min_beta1 >= 2 * np.pi) or (tr[0] == -np.pi and tr[1] == np.pi):
                max_beta1 = np.pi
                min_beta1 = -np.pi
    ion2 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta2_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[1][0] >= tr[0] and CA[1][1] <= tr[1]:
            ion2 = True
            # print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                # print('joint 2 succeed')

    ion3 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta3_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[2][0] >= tr[0] and CA[2][1] <= tr[1]:
            ion3 = True
            # print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                # print('joint 3 succeed')

    ion4 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta4_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[3][0] >= tr[0] and CA[3][1] <= tr[1]:
            ion4 = True
            # print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                # print('joint 4 succeed')
    if single_intersection_tf_all:F_list[-1] = True
    if ion1:
        if single_intersection_tf_all: F_list[0] = True
    if ion1 and ion2:
        if single_intersection_tf_all: F_list[4] = True
    if ion1 and ion3:
        if single_intersection_tf_all: F_list[5] = True
    if ion1 and ion4:
        if single_intersection_tf_all: F_list[6] = True
    if ion1 and ion2 and ion3:
        if single_intersection_tf_all: F_list[10] = True
    if ion1 and ion2 and ion4:
        if single_intersection_tf_all: F_list[11] = True
    if ion1 and ion3 and ion4:
        if single_intersection_tf_all: F_list[12] = True
    if ion1 and ion2 and ion3 and ion4:
        if single_intersection_tf_all: F_list[14] = True
    if ion2:
        if single_intersection_tf_all: F_list[1] = True
    if ion3:
        if single_intersection_tf_all: F_list[2] = True
    if ion4:
        if single_intersection_tf_all: F_list[3] = True
    if ion2 and ion3:
        if single_intersection_tf_all: F_list[7] = True
    if ion2 and ion4:
        if single_intersection_tf_all: F_list[8] = True
    if ion3 and ion4:
        if single_intersection_tf_all: F_list[9] = True
    if ion2 and ion3 and ion4:
        if single_intersection_tf_all: F_list[13] = True


    for index in range(len(beta0_ranges)):
        reliable_beta_ranges[-1].append(beta0_ranges[index])
        min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
        min_beta_f_ftw_v1 = max(min_beta0, -np.pi)
        max_beta_f_ftw_v1 = min(max_beta0, np.pi)
        v1_valid = False
        valid = False
        if min_beta_f_ftw_v1 <= max_beta_f_ftw_v1: v1_valid = True
        min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
        max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
        if min_beta_f_ftw <= max_beta_f_ftw: valid = True
        #print(ion1,ion2,ion3,ion4)
        if valid:
            if ion1:

                reliable_beta_ranges[0].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2:

                reliable_beta_ranges[4].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion3:

                reliable_beta_ranges[5].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion4:

                reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion3:

                reliable_beta_ranges[10].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[0].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion4:

                reliable_beta_ranges[11].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[1].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion3 and ion4:

                reliable_beta_ranges[12].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[2].append([min_beta_f_ftw, max_beta_f_ftw])
            if ion1 and ion2 and ion3 and ion4:

                reliable_beta_ranges[14].append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[4].append([min_beta_f_ftw, max_beta_f_ftw])

        if v1_valid:
            if ion2:

                reliable_beta_ranges[1].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion3:

                reliable_beta_ranges[2].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion4:

                reliable_beta_ranges[3].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion3:

                reliable_beta_ranges[7].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion4:

                reliable_beta_ranges[8].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion3 and ion4:

                reliable_beta_ranges[9].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
            if ion2 and ion3 and ion4:

                reliable_beta_ranges[13].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
                # reliable_beta_ranges[3].append([min_beta_f_ftw_v1, max_beta_f_ftw_v1])
    for i in range(16):
        reliable_beta_ranges[i] = union_ranges(reliable_beta_ranges[i])
        all_reliable_beta_ranges[i].append(reliable_beta_ranges[i])
    # for i in range(5):
    #    reliable_beta_ranges[i] = union_ranges(reliable_beta_ranges[i])
    #    all_reliable_beta_ranges[i].append(reliable_beta_ranges[i])
    return all_reliable_beta_ranges,F_list


"""
@measure_time
def ssm_estimation(grid_sample_num, d, alpha, l, CA):
    C_dot_A = CA.copy()
    C_dot_A[0] = (-np.pi, np.pi)
    C_dot_A = convert_to_C_dot_A(C_dot_A)
    robot = DHRobot(
        [
            RevoluteDH(d=d[0], alpha=alpha[0], a=l[0], qlim=CA[0]),
            RevoluteDH(d=d[1], alpha=alpha[1], a=l[1], qlim=CA[1]),
            RevoluteDH(d=d[2], alpha=alpha[2], a=l[2], qlim=CA[2]),
            RevoluteDH(d=d[3], alpha=alpha[3], a=l[3], qlim=CA[3]),
        ], name="spatial 4R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(4):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_size = (64, 64, 64)
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
    angle_ranges = []
    reachable_points = 0

    for center in tqdm(grid_centers , desc="Processing Items"):
        print(center)
        beta_ranges = compute_beta_range(center[0], center[1], center[2], robot, C_dot_A, CA)
        # print(beta_ranges)
        if len(beta_ranges) != 0: reachable_points += 1
        angle_ranges.append(beta_ranges)
    
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Generate grid of squares
    arc_color = 'blue'
    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Draw arcs for each square grid by rotating the entire grid square、
    for i, square in enumerate(grid_squares):
        for beta_range in angle_ranges[i]:
            draw_rotated_grid(ax, square, beta_range, arc_color)

    # Set plot labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    #plot sampled grids
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    # Draw squares only if the angle_ranges[i] is non-empty
    for i, square in enumerate(grid_squares):
        color = 'k'
        alpha_level = 0.3
        if angle_ranges[i]:  # Check if the list is non-empty
            alpha_level = 1.0
            # Plot the square grid directly
        square_poly = Poly3DCollection([square], color=color, alpha=alpha_level)
        ax.add_collection3d(square_poly)

    frame_points = [
        (x_range[0], 0, z_range[0]), (x_range[1], 0, z_range[0]),
        (x_range[1], 0, z_range[1]), (x_range[0], 0, z_range[1]),
        (x_range[0], 0, z_range[0])  # Closing the loop
    ]

    frame = Line3DCollection([frame_points], colors='k', linewidths=2)
    ax.add_collection3d(frame)

    # Set plot labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
        n_x, n_z, x_range, z_range, grid_size, angle_ranges
    )

    # grayscale_matrix = convert_plot_to_voxel_matrix(ax)
    shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_matrix,
                                                                                     kernel_size, Lambda)
    # print(shape_area)
    # print(connected_connectivity)
    print(general_connectivity)
    return general_connectivity
"""


# reliability
# @measure_time
def ssm_estimation(grid_sample_num, d, alpha, l, CA):
    C_dot_A = CA.copy()
    C_dot_A[0] = (-np.pi, np.pi)
    C_dot_A = convert_to_C_dot_A(C_dot_A)
    robot = DHRobot(
        [
            RevoluteDH(d=d[0], alpha=alpha[0], a=l[0], qlim=CA[0]),
            RevoluteDH(d=d[1], alpha=alpha[1], a=l[1], qlim=CA[1]),
            RevoluteDH(d=d[2], alpha=alpha[2], a=l[2], qlim=CA[2]),
            RevoluteDH(d=d[3], alpha=alpha[3], a=l[3], qlim=CA[3]),
        ], name="spatial 4R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(4):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    print(max_length)
    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_size = (64, 64, 64)
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)

    """uniform sample
    cells_per_radius = 4  # so cube side length d = R/16
    side_length = max_length / cells_per_radius

    # Edges and centers along one axis
    edges = np.arange(-max_length, max_length + 1e-12, side_length)
    n_cells = edges.size - 1
    centers = np.linspace(-max_length + side_length / 2, max_length - side_length / 2, n_cells)

    # 3D grid of cube centers
    x_c, y_c, z_c = np.meshgrid(centers, centers, centers, indexing='ij')

    # Keep centers inside the sphere of radius R
    in_sphere = (x_c**2 + y_c**2 + z_c**2) <=max_length**2

    cut3 = ((x_c >= 0) & (y_c >= 0) & (z_c >= 0)) | \
           ((x_c <= 0) & (y_c >= 0) & (z_c <= 0)) | \
           ((x_c <= 0) & (y_c <= 0) & (z_c >= 0)) | \
           ((x_c <= 0) & (y_c >= 0) & (z_c >= 0))

    mask = in_sphere & (~cut3)
    #mask = in_sphere

    x_values = x_c[mask]
    y_values = y_c[mask]
    z_values = z_c[mask]

    grid_centers  = np.column_stack((x_values, y_values, z_values))


    print("cube_side_length d =", side_length)
    print("cells per axis =", n_cells)
    print("num cubes selected =", len(grid_centers))
    """


    """random sample
    N = 17256

    def sample_in_sphere(n, r, seed=None):
        rng = np.random.default_rng(seed)
        out = np.empty((n, 3), dtype=float)
        filled = 0
        r2 = r * r
        batch = max(4096, int(n * 1.2))

        while filled < n:
            pts = rng.uniform(-r, r, size=(batch, 3))  # totally random x,y,z in cube
            in_sphere = np.einsum('ij,ij->i', pts, pts) <= r2  # keep those inside sphere
            kept = pts[in_sphere]
            take = min(kept.shape[0], n - filled)
            if take:
                out[filled:filled + take] = kept[:take]
                filled += take
            if kept.shape[0] < batch // 4 and batch < 1_000_000:
                batch *= 2
        return out

    # 1) sample exactly 17,256 points inside the sphere
    points = sample_in_sphere(N, max_length, seed=None)  # set seed=int if you want reproducibility
    x_c, y_c, z_c = points.T

    # 2) mask for the four octants to EXCLUDE (your ranges)
    cut4 = ((x_c >= 0) & (y_c >= 0) & (z_c >= 0)) | \
           ((x_c <= 0) & (y_c >= 0) & (z_c <= 0)) | \
           ((x_c <= 0) & (y_c <= 0) & (z_c >= 0)) | \
           ((x_c <= 0) & (y_c >= 0) & (z_c >= 0))

    # 3) valid points are NOT in those ranges
    valid_mask = ~cut4
    grid_centers = points[valid_mask]
    #grid_centers=points
    side_length=max_length / 16.0

    print("Sampled inside sphere:", N)
    print("Final number of valid points:", grid_centers.shape[0])
    """




    all_reliable_beta_ranges = [[] for _ in range(16)]



    #test_center=[1.5,0,-0.2]
    #all_reliable_beta_ranges = compute_reliable_beta_range(test_center[0],test_center[1],test_center[2], robot, C_dot_A, CA,
     #                                                      all_reliable_beta_ranges)
    indices = sorted_indices(cr_list)
    cr_list.append(r1*r2*r3*r4/(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4 - 3 * r1 * r2 * r3 * r4))
    color_list, sm = normalize_and_map_colors(cr_list)
    """ uniform and random
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    """

    # """original
    fig, ax2 = plt.subplots()
    ax2.set_xlim([0, max_length])
    ax2.set_ylim([-max_length, max_length])
    ax2.set_aspect(1)
    cbar = plt.colorbar(sm, ax=ax2, label='Reliability Spectrum')

    plt.ion()
    index_dict = {}
    #"""original




    start = time.perf_counter()
    twod_squares = generate_2D_square_grid(n_x, n_z, x_range, z_range)
    for i, center in tqdm(enumerate(grid_centers), total=len(grid_centers), desc="Processing grid centers"):
        # print(center)
        square = twod_squares[i]
        all_reliable_beta_ranges, F_list = compute_reliable_beta_range(
            center[0], center[1], center[2], robot, C_dot_A, CA, all_reliable_beta_ranges
        )

        last_true = None

        if F_list[-1]:
            last_true=-1
        for you in indices:  # indices is ascending
            if F_list[you]:
                last_true = you

        if last_true is not None:
            color = color_list[last_true]
            update_or_add_square_2d(ax2, square, color, 1.0, i, index_dict=index_dict) #original
            # draw_cube(ax, center, side_length, color)  #uniform
            # draw_sphere(ax, center, side_length, color) #random

    end = time.perf_counter()
    print(f"Loop took {end - start:.6f} seconds")


    #grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    with open("my_list.txt", "w") as file:
        file.write(str(all_reliable_beta_ranges))

    """ uniform and random
    R = max_length
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    set_axes_equal(ax)

    ax.set_box_aspect([1, 1, 1])  # forces equal aspect

    #ax.view_init(elev=28, azim=-35)  # try (28, -35); tweak ±2° if needed
    #ax.set_proj_type('persp')
    #ax.dist = 9
    axis_length = max_length * 1.2  # extend beyond the shape for visibility
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # X-axis
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Y-axis
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Z-axis

    # Add axis labels near arrow tips
    ax.text(axis_length, 0, 0, r"$x$", fontsize=14, color='k')
    ax.text(0, axis_length, 0, r"$y$", fontsize=14, color='k')
    ax.text(0, 0, axis_length, r"$z$", fontsize=14, color='k')

    ax.set_xlabel("x", fontsize=25)
    ax.set_ylabel("y", fontsize=25)
    ax.set_zlabel("z", fontsize=25)
    tick_positions = [-3,-2, -1, 0, 1, 2,3]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_zticks(tick_positions)
    ax.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.view_init(elev=35, azim=146, roll=-111)
    plt.show()
    """




    #original extra beta
    fig, ax = plt.subplots(figsize=(12, 4))

    sub_indices = [
        i for i in range(len(grid_centers))
        if any(len(all_reliable_beta_ranges[you][i]) != 0 for you in indices)
    ]
    sub_grid_centers = [grid_centers[i] for i in sub_indices]



                # Optional placeholders (just to show columns exist)
    for i in range(len(grid_centers)):
        ax.bar(i, 0, width=0.6, color="white", edgecolor="purple")

    # Style (like before)
    tick_positions = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=18)
    #ax.set_xticks(range(len(grid_centers)))
    #ax.set_xticklabels([f"({x:.2f},{z:.2f})" for x,y,z in grid_centers], rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Failure Tolerant Rotation Angles of β", fontsize=27)
    ax.set_xlabel("x, z Task Space Locations", fontsize=27)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    NDIGITS = 8
    #x_index_map = {key3(pt, ndigits=NDIGITS): i for i, pt in enumerate(grid_centers)}
    #x_index_map = {key3(pt, ndigits=NDIGITS): i for i, pt in enumerate(sub_grid_centers)}
    x_index_map = {
        key3(grid_centers[i], ndigits=NDIGITS): j
        for j, i in enumerate(sub_indices)
    }
    #original"""

    #"""original3D
    fig = plt.figure()
    ax4 = fig.add_subplot(111, projection='3d')

    ax4.set_xlim([-3, 3])  # todo: optimized
    ax4.set_ylim([-3, 3])
    ax4.set_zlim([-3, 3])
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    #"""

    # """orignal 3D
    arc_color = color_list[-1]
    for i, square in tqdm(enumerate(grid_squares), desc="Processing Items"):
        for beta_range in all_reliable_beta_ranges[-1][i]:
            # draw_wedge(ax, square, beta_range, arc_color)
            draw_rotated_grid(ax4, square, beta_range, arc_color)

    # """

    for you in indices:
        ftw_points_count = 0

        #""" original 3D
        arc_color = color_list[you]

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #ax.set_xlim([-3, 3])  # todo: optimized
        #ax.set_ylim([-3, 3])
        #ax.set_zlim([-3, 3])

        for i, square in tqdm(enumerate(grid_squares), desc="Processing Items"):
            for beta_range in all_reliable_beta_ranges[you][i]:
                # draw_wedge(ax, square, beta_range, arc_color)
                draw_rotated_grid(ax4, square, beta_range, arc_color)

        #"""
        #ax.view_init(elev=30, azim=135)

        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.draw()
        #print("Press 'q' to continue...")
        #while True:
        #    key = plt.waitforbuttonpress()
        #    if key:  # Any key will work, but we can restrict it if needed
        #       break

        #plt.close(fig)
        #"""
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

        """random and uniform
        twod_squares = generate_2D_square_grid(n_x, n_z, x_range, z_range)
        for i, square in enumerate(twod_squares):
            #color = 'w'
            #alpha_level=1.0
            if all_reliable_beta_ranges[you][i]:
                #color = color_list[you]
                ftw_points_count += 1
        """
        # """original
        for j, i in enumerate(sub_indices):
            pt = grid_centers[i]
            ranges = all_reliable_beta_ranges[you][i]
            if not ranges:
                continue
            update_beta_bar_multicolor(
                ax, pt, ranges,
                zorder=you + 1,
                x_index_map=x_index_map,  # built from the same pts
                ndigits=NDIGITS
            )
        #original"""

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



    #"""orignal 3D
    ax4.view_init(elev=35, azim=146, roll=-111)
    R = max_length
    ax4.set_xlim(-R, R)
    ax4.set_ylim(-R, R)
    ax4.set_zlim(-R, R)
    set_axes_equal(ax4)
    ax4.set_box_aspect([1, 1, 1])  # forces equal aspect
    axis_length = max_length * 1.2  # extend beyond the shape for visibility
    ax4.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # X-axis
    ax4.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Y-axis
    ax4.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # Z-axis

    # Add axis labels near arrow tips
    ax4.text(axis_length, 0, 0, r"$x$", fontsize=14, color='k')
    ax4.text(0, axis_length, 0, r"$y$", fontsize=14, color='k')
    ax4.text(0, 0, axis_length, r"$z$", fontsize=14, color='k')
    ax4.set_xlabel("x", fontsize=25)
    ax4.set_ylabel("y", fontsize=25)
    ax4.set_zlabel("z", fontsize=25)
    tick_positions = [-3, -2, -1, 0, 1, 2, 3]
    ax4.set_xticks(tick_positions)
    ax4.set_yticks(tick_positions)
    ax4.set_zticks(tick_positions)
    ax4.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax4.tick_params(axis='y', labelsize=18)
    ax4.tick_params(axis='z', labelsize=18)
    plt.show()
    #"""



    """
    plt.show()
    print("Press 'q' to continue...")
    while True:
        key = plt.waitforbuttonpress()
        if key:  # Any key will work, but we can restrict it if needed
            break

    plt.close(fig)
    """


    #"""original
    set_sparse_xyz_labels(ax, sub_grid_centers, max_labels=20, ndigits=2)
    ax.set_xlim(-0.5, len(sub_grid_centers) - 0.5)
    ax.set_facecolor("white")
    ax2.set_xlabel('X', fontsize=24)
    ax2.set_ylabel('Z', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=18)


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
    reliable_connectivity = 0

    for index, angle_ranges in enumerate(all_reliable_beta_ranges):
        binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
            n_x, n_z, x_range, z_range, grid_size, angle_ranges
        )

        connected_connectivity = connectivity_analysis(binary_matrix,kernel_size, Lambda)
        reliable_connectivity += cr_list[index] * np.sum(binary_matrix)*connected_connectivity
    print(f' The general reliable connectivity considering top 5 cases is{reliable_connectivity}.')
    return reliable_connectivity
    #original"""


CA = [(-146 * np.pi / 180, 146 * np.pi / 180), (-234 * np.pi / 180, 10 * np.pi / 180),
     (-115 * np.pi / 180, 132 * np.pi / 180), (-101 * np.pi / 180, 118 * np.pi / 180)]
#CA=[(-1.1462323775013838, 3.0555459324329153), (-1.1618976450230296, 3.141592653589793), (-2.690561890475438, 1.9074680849495245), (-0.5390132397825012, 0.4414181348394396)]
# CA =  [(-0.34476583954363793, 0.34476583954363793), (-3.8557928335253506, 0.5727802075632915),
#                   (-2.0120387975312504, 0.8610682582634244), (-1.6048847508602293, 2.1951493670529656)]
# CA = [(-2.017801347479772, 2.017801347479772), (-3.0735622252855346, -1.8767802693000228), (-2.3001230516678093, -1.0323465272698482), (-1.091382218006809, 0.5810127777635778)]
#
#CA = [(-0.8730382117746103, 0.8730382117746103), (-3.07870404714172, 2.979105017877223),
#      (-2.965730966059918, 2.831650417251087), (0.6494635825661303, 2.9645940008572973)]
alpha = [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180]
d = [-0.29, 0, 0.05, 1]
l = [0.5, 0.48, 0.76, 0.95]
ap = ssm_estimation(512, d, alpha, l, CA)
print(ap)

# d = [-0.019917995106395026, 0.6118090376463043, 0.05065138908443867, 0.45487466192184756]
# alpha = [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180]
# alpha=  [0.7334761894150401, -0.7205303423799283, -1.3089320990376847, 1.5510841614806563]


# l =   [0.4678658670270923, 0.4484934743492972, 0.860979553181329, 0.832349768252797]
# ap = ssm_estimation(72, d, alpha, l, CA)

# 72 98 128 162 200 242 288 338 392 512
# alpha = [-0.5066408223966066, -1.1913266026059928, -0.06860434916763447, -0.4230013608940213]
# l = [0.8208970624546458, 0.17586976764525408, 0.9444232782190134, 0.638882188359731]
# d = [-0.19528312746924503, -0.7458185218572451, -0.5671355232019555, -0.6448212471876422]
# CA = [(-1.0164484727118497, 1.0164484727118497), (-2.9905042393967816, 0.942118647580509),
#     (0.5725921882509168, 1.4935540364595248), (0.1414561277832811, 2.3221616710024895)]
# ap = ssm_estimation(72, d, alpha, l, CA)

# l = [0.50, 0.48, 0.76, 0.95]
# l = [-0.29, 0, 0.05, 1.0]
# CA = [(-0.8730382117746103, 0.8730382117746103), (-3.07870404714172, 2.979105017877223), (-2.965730966059918, 2.831650417251087), (0.6494635825661303, 2.9645940008572973)]

# ap = ssm_estimation(5000, d, alpha, l, CA)
# print(beta_ranges)

# print(x)
# print(new_x)
# print(delta_x)
# print(x_next)
# print(theta)
# print(new_theta)
# print(delta_theta)
# (theta_next)
# print(J_plus)
