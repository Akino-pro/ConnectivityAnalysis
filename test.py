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
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 30
max_ssm = 4


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
    a = np.array(n_j).reshape(-1)  # Converts 4x1 to 1D array
    b = np.array(previous_n_j).reshape(-1)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    if np.abs(angle_rad) > np.pi / 2:
        n_j = -n_j

    x = robot.fkine(theta.flatten())
    x = np.array(x)[:3, 3].T.reshape((3, 1))
    delta_x_step = x - x_target
    J_step = robot.jacob0(theta.flatten())[:3, :]
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    theta_next = theta + step_size * n_j / np.linalg.norm(n_j) + corrected_delta_theta
    # print(theta_next)
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    x_next = robot.fkine(theta_next.flatten())
    x_next = np.array(x_next)[:3, 3].T.reshape((3, 1))
    new_J = robot.jacob0(theta_next.flatten())[:3, :]
    back_step = 1
    while np.linalg.norm(x_target - x_next) > 1e-6:
        back_step += 1
        new_J_plus = np.linalg.pinv(new_J)
        theta_next += np.dot(new_J_plus, x_target - x_next)
        for i in range(len(theta_next)):
            theta_next[i] %= 2 * np.pi
            if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
            if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
        x_next = robot.fkine(theta_next.flatten())
        x_next = np.array(x_next)[:3, 3].T.reshape((3, 1))
        new_J = robot.jacob0(theta_next.flatten())[:3, :]
    new_n_j = null_space(new_J)
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
        if np.abs(local_min_theta1 + np.pi) <= 1e-3: local_min_theta1 = -np.pi
        if np.abs(local_min_theta1 - np.pi) <= 1e-3: local_min_theta1 = np.pi
        if np.abs(local_max_theta1 + np.pi) <= 1e-3: local_max_theta1 = -np.pi
        if np.abs(local_max_theta1 - np.pi) <= 1e-3: local_max_theta1 = np.pi
        ip_ranges.append([local_min_theta1, local_max_theta1])
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
            return [[-2 * np.pi, 2 * np.pi]]
        elif tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        elif tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        else:
            extended_ranges.append(tr)
    if has_negative_pi_range is not None and has_positive_pi_range is not None:
        extended_ranges.append([has_positive_pi_range - 2 * np.pi, has_negative_pi_range])
        extended_ranges.append([has_positive_pi_range, has_negative_pi_range + 2 * np.pi])
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

    for current in ranges[1:]:
        last = merged[-1]

        # If the current range overlaps or touches the last merged range, merge them
        if current[0] <= last[1]:  # Overlap or adjacent
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # No overlap, add the current range as a new entry
            merged.append(current)
    return merged


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
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(4):
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
    # print(thetas_cp_sum)
    return thetas_cp_sum


def find_random_ssm(x_target, all_ssm_theta_list, robot, C_dot_A):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(4,))).T.reshape((4, 1))
    Tep = SE3(x_target)
    result = robot.ikine_LM(Tep, mask=[1, 1, 1, 0, 0, 0], q0=q)
    if not result.success: return result.success, [], [[], [], [], []], all_ssm_theta_list, ssm_found
    sol = result.q.T.reshape((4, 1))
    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return True,[], [[], [], [], []], all_ssm_theta_list, ssm_found
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
        if threshold <= terminate_threshold and num > 4: break
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j, robot)
        threshold = np.linalg.norm(theta - theta_prime)
        if threshold < lowest: lowest = threshold
        if num == 100:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm guided to wrong direction.')
                    return True,[], [[], [], [], []], all_ssm_theta_list, ssm_found
        n_j = new_n_j
        if num > 15000 and lowest >= terminate_threshold:
            print(robot)
            print(theta_prime)
            print('if you see this printed, then there are still spaces for efficiency improvement.')
            print('-------------------------------------------------------------------------------.')
            print('force terminate')
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

            theta_prime = theta
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm already exists.')
                    return True,[], [[], [], [], []], all_ssm_theta_list, ssm_found
            # return [], -10000, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, all_ssm_theta_list, ssm_found
            ssm_theta_list = [theta]
        if num > 20000: return True,[], [[], [], [], []], all_ssm_theta_list, ssm_found
        num += 1
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
    return True,ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found


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
        ik,iprs, cp_ranges, all_theta, ssm_found_tf = find_random_ssm(
            target_x, all_theta, robot, C_dot_A)
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)
        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
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
    """
    if len(all_theta) != 0 and x==1.25 and z==1.25:
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
    #print(beta0_ranges)
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
    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    has_negative_pi_range = None
    has_positive_pi_range = None
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if tr[0] == -np.pi:
            has_negative_pi_range = tr[1]
        if tr[1] == np.pi:
            has_positive_pi_range = tr[0]
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
            print('joint 1 succeed')
            min_beta1 = CA[0][1] - tr[1]
            max_beta1 = CA[0][0] - tr[0]
            if max_beta1 - min_beta1 >= 2 * np.pi:
                max_beta1 = np.pi
                min_beta1 = -np.pi
    if CA[0][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[0][1] <= has_negative_pi_range and CA[0][0] + 2 * np.pi >= has_positive_pi_range:
                ion1 = True
                print('joint 1 succeed')
                min_beta1 = CA[0][1] - has_negative_pi_range
                max_beta1 = CA[0][0] - has_positive_pi_range + 2 * np.pi
                if max_beta1 - min_beta1 >= 2 * np.pi:
                    max_beta1 = np.pi
                    min_beta1 = -np.pi
    if CA[0][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[0][1] - 2 * np.pi <= has_negative_pi_range and CA[0][0] >= has_positive_pi_range:
                ion1 = True
                print('joint 1 succeed')
                min_beta1 = CA[0][1] - has_negative_pi_range - 2 * np.pi
                max_beta1 = CA[0][0] - has_positive_pi_range
                if max_beta1 - min_beta1 >= 2 * np.pi:
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
            print('joint 2 succeed')
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
                print('joint 2 succeed')
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
                print('joint 2 succeed')
    min_beta2 = -np.pi
    max_beta2 = np.pi

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
            print('joint 3 succeed')
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
                print('joint 3 succeed')
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
                print('joint 3 succeed')
    min_beta3 = -np.pi
    max_beta3 = np.pi

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
            print('joint 4 succeed')
    if CA[3][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] <= has_negative_pi_range and CA[3][0] + 2 * np.pi >= has_positive_pi_range:
                ion4 = True
                print('joint 4 succeed')
    if CA[3][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[3][1] - 2 * np.pi <= has_negative_pi_range and CA[3][0] >= has_positive_pi_range:
                ion4 = True
                print('joint 4 succeed')
    min_beta4 = -np.pi
    max_beta4 = np.pi

    for index in range(len(beta0_ranges)):
        min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
        #print(min_beta0, max_beta0)
        #print(min_beta1, max_beta1)
        #print(min_beta2, max_beta2)
        #print(min_beta3, max_beta3)
        #print(min_beta3, max_beta4)

        if ion1 and ion2 and ion3 and ion4:
            min_beta_f_ftw = max(min_beta0, min_beta1, min_beta2, min_beta3, min_beta4)
            max_beta_f_ftw = min(max_beta0, max_beta1, max_beta2, max_beta3, max_beta4)

            if min_beta_f_ftw <= max_beta_f_ftw:
                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
                # reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])

    # for rbr_index in range(len(reliable_beta_ranges)):
    #    reliable_beta_ranges[rbr_index] = union_ranges(reliable_beta_ranges[rbr_index])
    # print(all_smm_beta_range)
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
        ], name="spatial 4R")
    N = grid_sample_num
    n_z = int(np.sqrt(2 * N))
    n_x = int(n_z / 2)  # Number of grid divisions along z-axis
    x_range = (0, 4)  # Range for x-axis
    z_range = (-4, 4)  # Range for z-axis
    grid_size = (32, 32, 32)
    grid_centers = generate_grid_centers(n_x, n_z, N)
    # print("3D coordinates of center points:")
    angle_ranges = []
    for center in grid_centers:
        # for center in tqdm(grid_centers, desc="Processing centers"):
        # Compute beta ranges for each center
        print(center)
        beta_ranges = compute_beta_range(center[0], center[1], center[2], robot, C_dot_A, CA)
        print(beta_ranges)
        angle_ranges.append(beta_ranges)

    # Generate grid of squares
    grid_squares = generate_square_grid(n_x, n_z, x_range, z_range)
    arc_color = 'blue'

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot range
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    # Draw arcs for each square grid by rotating the entire grid square
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


CA = [(-146 * np.pi / 180, 146 * np.pi / 180), (-180 * np.pi / 180, 10 * np.pi / 180),
      (-115 * np.pi / 180, 132 * np.pi / 180), (-101 * np.pi / 180, 118 * np.pi / 180)]

# CA =[(-2.9774403115370007, 2.9774403115370007), (-3.649806052899172, 2.9243943979042584), (-1.1932044930661232, -0.7231275709033502), (-2.6463678609846166, -1.4370950210363702)]
d = [-0.29, 0, 0.05, 1]
# d = [0.18965404131623143, 0.8611567328006011, 0.418680786709572, 0.5984359880735328]
alpha = [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180]
# alpha=  [0.4550792062494238, 1.393655250013011, 1.5084831068876752, -0.6023770982898573]
l = [0.5, 0.48, 0.76, 0.95]

# a = [0.18965404131623143, 0.8611567328006011, 0.418680786709572, 0.5984359880735328]
# ap = ssm_estimation(sample_num, d, alpha, a, CA)

# alpha = [-0.5066408223966066, -1.1913266026059928, -0.06860434916763447, -0.4230013608940213]
# a = [0.8208970624546458, 0.17586976764525408, 0.9444232782190134, 0.638882188359731]
# d = [-0.19528312746924503, -0.7458185218572451, -0.5671355232019555, -0.6448212471876422]
# CA = [(-1.0164484727118497, 1.0164484727118497), (-2.9905042393967816, 0.942118647580509),
# (0.5725921882509168, 1.4935540364595248), (0.1414561277832811, 2.3221616710024895)]
ap = ssm_estimation(72, d, alpha, l, CA)
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
