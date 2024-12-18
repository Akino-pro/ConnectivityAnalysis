import time

import numpy as np
from tqdm import tqdm
from scipy.linalg import null_space
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3

from spatial3R_ftw_draw import generate_grid_centers, generate_square_grid, draw_rotated_grid
from Three_dimension_connectivity_measure import connectivity_analysis
from spatial3R_ftw_draw import generate_binary_matrix

kernel_size = 1
Lambda = 0.5
step_size = 0.01
#terminate_threshold = 5.0 * step_size
terminate_threshold = 9.0/5.0 * step_size
#terminate_threshold = step_size * 0.5
ssm_finding_num = 20
max_ssm = 16


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


def stepwise_ssm(theta, n_j, Tep, previous_n_j, robot):
    x_pos_ori = robot.fkine(theta.flatten())
    x_pos = np.array(x_pos_ori)[:3, 3].T.reshape((3, 1))
    x_target = np.array(Tep)[:3, 3].T.reshape((3, 1))
    cn = np.array(x_pos_ori)[:3, 0].T.reshape((3, 1))
    co = np.array(x_pos_ori)[:3, 1].T.reshape((3, 1))
    ca = np.array(x_pos_ori)[:3, 2].T.reshape((3, 1))
    tn = np.array(Tep)[:3, 0].T.reshape((3, 1))
    to = np.array(Tep)[:3, 1].T.reshape((3, 1))
    ta = np.array(Tep)[:3, 2].T.reshape((3, 1))
    delta_pos = x_target - x_pos
    delta_ori_n = np.cross(tn.flatten(), cn.flatten())
    delta_ori_o = np.cross(to.flatten(), co.flatten())
    delta_ori_a = np.cross(ta.flatten(), ca.flatten())
    delta_ori = 0.5 * (delta_ori_n + delta_ori_o + delta_ori_a).reshape((3, 1))
    delta_x_step = np.vstack((delta_pos, delta_ori))

    # J_step = robot.jacob0(theta.flatten())[:3, :]
    J_step = robot.jacob0(theta.flatten())
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + dq * step_size
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    # new_J = robot.jacob0(theta_next.flatten())[:3, :]
    new_J = robot.jacob0(theta_next.flatten())
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


def extend_ranges(union_ranges):
    extended_ranges = []
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in union_ranges:
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


def find_critical_points(ssm_theta_list):
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf],
                       [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], [], [], [], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
                , [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
                , [ssm_theta_list[0][4][0], ssm_theta_list[0][4][0]]
                , [ssm_theta_list[0][5][0], ssm_theta_list[0][5][0]]
                , [ssm_theta_list[0][6][0], ssm_theta_list[0][6][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        thetas_cp_range[3] = [ssm_theta_list[0][3][0], ssm_theta_list[0][3][0]]
        thetas_cp_range[4] = [ssm_theta_list[0][4][0], ssm_theta_list[0][4][0]]
        thetas_cp_range[5] = [ssm_theta_list[0][5][0], ssm_theta_list[0][5][0]]
        thetas_cp_range[6] = [ssm_theta_list[0][6][0], ssm_theta_list[0][6][0]]
        previous_theta = ssm_theta_list[0]
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(7):
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
    for j in range(7):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
    # print(thetas_cp_sum)
    return thetas_cp_sum


def find_random_ssm(x_target, all_ssm_theta_list, robot, C_dot_A):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(7,))).T.reshape((7, 1))
    Tep = SE3(x_target)
    result = robot.ikine_LM(Tep, mask=[1, 1, 1, 0, 0, 0], q0=q)
    if not result.success: return result.success, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found
    sol = result.q.T.reshape((7, 1))
    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found
    theta = sol
    theta_prime = theta.copy()

    # J = robot.jacob0(theta.flatten())[:3, :]
    J = robot.jacob0(theta.flatten())
    n_j = null_space(J)[:, 0].T.reshape((7, 1))
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    print(terminate_threshold)
    while True:
        num += 1
        if threshold <= terminate_threshold and num >= 4: break
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, Tep, old_n_j, robot)
        threshold = np.linalg.norm(theta - theta_prime)
        # print(threshold)
        if threshold < lowest: lowest = threshold

        if num == 100:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True, [], [[], [], [], [], [], [], []], all_ssm_theta_list, ssm_found
        # m, n = np.shape(new_n_j)
        n_j = new_n_j
        # print('new n_j')
        # print(n_j)
        # if n == 2:
        #   print('special case')
        #    n_j_1 = new_n_j[:, 0]
        #    n_j_2 = new_n_j[:, 1]
        #    if np.dot(n_j_1, old_n_j) > np.dot(n_j_2, old_n_j):
        #        n_j = n_j_1
        #    else:
        #        n_j = n_j_2
        # if num > 10000: print('here')
        if num > 5000: print(threshold)
        if num > 10000:
            print('stuck at a too small smm')
            # for configuration in all_ssm_theta_list:
            #    if np.linalg.norm(configuration - theta) <= terminate_threshold:
            #        # print('ssm guided to wrong direction.')
            #        return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found
            # print('still not found')
            theta_prime = theta
            num = 0
            # return True, [], [[], [], [], []], all_ssm_theta_list, ssm_found
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
    print(f'found a new ssm with {num} points.')
    ssm_found = True

    ip_ranges, tof = find_intersection_points(ssm_theta_list, C_dot_A)
    cp_ranges = find_critical_points(ssm_theta_list)
    return True, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found


def compute_beta_range(x, y, z, robot, C_dot_A, CA):
    target_x = np.array([x, y, z]).T.reshape((3, 1))
    all_smm_beta_range = []
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    theta4_ranges = []
    theta5_ranges = []
    theta6_ranges = []
    theta7_ranges = []
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
        for cp_range in cp_ranges[4]:
            if cp_range[0] > -np.inf: theta5_ranges.append(cp_range)
        for cp_range in cp_ranges[5]:
            if cp_range[0] > -np.inf: theta6_ranges.append(cp_range)
        for cp_range in cp_ranges[6]:
            if cp_range[0] > -np.inf: theta7_ranges.append(cp_range)

    # print(beta0_ranges)
    if len(theta1_ranges) == 0: return []
    if len(theta2_ranges) == 0: return []
    if len(theta3_ranges) == 0: return []
    if len(theta4_ranges) == 0: return []
    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    theta4_ranges_union = union_ranges(theta4_ranges)
    theta5_ranges_union = union_ranges(theta5_ranges)
    theta6_ranges_union = union_ranges(theta6_ranges)
    theta7_ranges_union = union_ranges(theta7_ranges)

    # print(theta1_ranges_union)
    # print(theta2_ranges_union)
    # print(theta3_ranges_union)
    # print(theta4_ranges_union)
    # print('hello')
    # print(extend_ranges(theta1_ranges_union))
    # print(extend_ranges(theta2_ranges_union))
    # print(extend_ranges(theta3_ranges_union))
    # print(extend_ranges(theta4_ranges_union))
    """
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
    """
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

    ion5 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta5_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[4][0] >= tr[0] and CA[4][1] <= tr[1]:
            ion5 = True
            # print('joint 5 succeed')
    if CA[4][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[4][1] <= has_negative_pi_range and CA[4][0] + 2 * np.pi >= has_positive_pi_range:
                ion5 = True
                # print('joint 5 succeed')
    if CA[4][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[4][1] - 2 * np.pi <= has_negative_pi_range and CA[4][0] >= has_positive_pi_range:
                ion5 = True
                # print('joint 5 succeed')

    ion6 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta6_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[5][0] >= tr[0] and CA[5][1] <= tr[1]:
            ion6 = True
            # print('joint 6 succeed')
    if CA[5][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[5][1] <= has_negative_pi_range and CA[5][0] + 2 * np.pi >= has_positive_pi_range:
                ion6 = True
                # print('joint 6 succeed')
    if CA[5][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[5][1] - 2 * np.pi <= has_negative_pi_range and CA[5][0] >= has_positive_pi_range:
                ion6 = True
                # print('joint 6 succeed')

    ion7 = False
    has_negative_pi_range = None
    has_positive_pi_range = None
    for tr in theta7_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[6][0] >= tr[0] and CA[6][1] <= tr[1]:
            ion7 = True
            # print('joint 7 succeed')
    if CA[6][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[6][1] <= has_negative_pi_range and CA[6][0] + 2 * np.pi >= has_positive_pi_range:
                ion7 = True
                # print('joint 7 succeed')
    if CA[6][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[6][1] - 2 * np.pi <= has_negative_pi_range and CA[6][0] >= has_positive_pi_range:
                ion7 = True
                # print('joint 7 succeed')
    # for index in range(len(beta0_ranges)):
    #    min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
    #    min_beta_f_ftw = -np.pi
    #   max_beta_f_ftw = np.pi
    #    all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
    if ion1 and ion2 and ion3 and ion4 and ion5 and ion6 and ion7:
        for index in range(len(beta0_ranges)):
            min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]

            min_beta_f_ftw = max(min_beta0, min_beta1, -np.pi)
            max_beta_f_ftw = min(max_beta0, max_beta1, np.pi)
            if min_beta_f_ftw <= max_beta_f_ftw:
                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
    return union_ranges(all_smm_beta_range)


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
            RevoluteDH(d=d[4], alpha=alpha[4], a=l[4], qlim=CA[4]),
            RevoluteDH(d=d[5], alpha=alpha[5], a=l[5], qlim=CA[5]),
            RevoluteDH(d=d[6], alpha=alpha[6], a=l[6], qlim=CA[6]),
        ], name="spatial 7R")
    # print(np.sum(l))
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    max_length = 0
    for i in range(7):
        max_length += np.sqrt(np.power(d[i], 2) + np.power(l[i], 2))
    x_range = (0, max_length)  # Range for x-axis
    z_range = (-max_length, max_length)  # Range for z-axis
    grid_size = (64, 64, 64)
    grid_centers = generate_grid_centers(n_x, n_z, N, x_range, z_range)
    # print("3D coordinates of center points:")
    angle_ranges = []
    reachable_points = 0

    for center in grid_centers:
        # for center in tqdm(grid_centers, desc="Processing centers"):
        # Compute beta ranges for each center
        print(center)
        beta_ranges = compute_beta_range(center[0], center[1], center[2], robot, C_dot_A, CA)
        # print(beta_ranges)
        if len(beta_ranges) != 0: reachable_points += 1
        angle_ranges.append(beta_ranges)
    # print(reachable_points)

    # Generate grid of squares
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    arc_color = 'blue'
    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    #    for i, square in enumerate(grid_squares):
    #   center = grid_centers[i]
    #    if center[2] > 0:  # Upper half, keep only one quarter
    #        for beta_range in angle_ranges[i]:
    #            draw_rotated_grid(ax, square, [0.25 * beta_range[0], 0.25 * beta_range[1]], arc_color)
    #    else:  # Lower half, remove one quarter
    #        for beta_range in angle_ranges[i]:
    #            draw_rotated_grid(ax, square, [0.75 * beta_range[0], 0.75 * beta_range[1]], arc_color)

    # Draw arcs for each square grid by rotating the entire grid square、
    for i, square in enumerate(grid_squares):
        for beta_range in angle_ranges[i]:
            draw_rotated_grid(ax, square, beta_range, arc_color)

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
    return general_connectivity


"""
CA = [(-146 * np.pi / 180, 146 * np.pi / 180), (-234 * np.pi / 180, 10 * np.pi / 180),
      (-115 * np.pi / 180, 132 * np.pi / 180), (-101 * np.pi / 180, 118 * np.pi / 180)]
# CA = [(-np.pi, np.pi), (-np.pi, np.pi),
#      (-np.pi, np.pi), (-np.pi, np.pi)]
# CA = [(-2.017801347479772, 2.017801347479772), (-3.0735622252855346, -1.8767802693000228), (-2.3001230516678093, -1.0323465272698482), (-1.091382218006809, 0.5810127777635778)]
# CA =[(-2.9774403115370007, 2.9774403115370007), (-3.649806052899172, 2.9243943979042584), (-1.1932044930661232, -0.7231275709033502), (-2.6463678609846166, -1.4370950210363702)]
d = [-0.29, 0, 0.05, 1]
# d = [-0.019917995106395026, 0.6118090376463043, 0.05065138908443867, 0.45487466192184756]
alpha = [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180]
# alpha=  [0.7334761894150401, -0.7205303423799283, -1.3089320990376847, 1.5510841614806563]
l = [0.5, 0.48, 0.76, 0.95]

# l =   [0.4678658670270923, 0.4484934743492972, 0.860979553181329, 0.832349768252797]
ap = ssm_estimation(512, d, alpha, l, CA)
"""
# 72 98 128 162 200 242 288 338 392
alpha = [-62 * np.pi / 180, -79 * np.pi / 180, 90 * np.pi / 180, 29 * np.pi / 180, 81 * np.pi / 180, -80 * np.pi / 180,
         -90 * np.pi / 180]
# l = [0.8208970624546458, 0.17586976764525408, 0.9444232782190134, 0.638882188359731]
# d = [-0.19528312746924503, -0.7458185218572451, -0.5671355232019555, -0.6448212471876422]
# CA = [(-1.0164484727118497, 1.0164484727118497), (-2.9905042393967816, 0.942118647580509),
#     (0.5725921882509168, 1.4935540364595248), (0.1414561277832811, 2.3221616710024895)]
# ap = ssm_estimation(72, d, alpha, l, CA)
# alpha = [-1.1809311206221138, 0.776922502833985, -0.5242992337964443, 0.8058636927088405]
l = [0.4, 0.8, 0.2, 1, 0.6, 0.4, 0.2]
d = [-0.4, -0.6, 0.2, 0.6, -0.8, 0.2, 0.8]
CA = [(-107 * np.pi / 180, 107 * np.pi / 180), (-164 * np.pi / 180, 141 * np.pi / 180),
      (-132 * np.pi / 180, 132 * np.pi / 180), (-151 * np.pi / 180, 102 * np.pi / 180),
      (-115 * np.pi / 180, 149 * np.pi / 180), (-75 * np.pi / 180, 129 * np.pi / 180),
      (16 * np.pi / 180, 193 * np.pi / 180)]
ap = ssm_estimation(512, d, alpha, l, CA)
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
