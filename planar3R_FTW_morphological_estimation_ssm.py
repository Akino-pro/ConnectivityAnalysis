import json
import math
import time

import numpy as np
from matplotlib import patches
from matplotlib.patches import Wedge, Rectangle, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from helper_functions import normalize_and_map_colors, sorted_indices, wedge_polygon, plot_exterior_boundary, \
    sample_line, normalize_and_map_greyscale

step_size = 0.01
# terminate_threshold = step_size / 2.0
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 20
max_ssm = 2
sample_num = 100
#r1 =0.3
#r2 =0.2
#r3 =0.1
r1 = 1.0/3.0
r2 = 1.0/3.0
r3 = 1.0/3.0

# L = [0.4888656043245976, 1.3992499610293656, 1.1118844346460368]
# L = [1, 1, 1]
# L = [0.009651087409352832, 1.6279980832723875, 1.3623508293182596]
# CA = [(-2.312033825326607, 2.312033825326607), (-2.1986530478299358, 1.3083974620434837),
#     (-3.119803865520389, 0.38031991292340495)]
# L=[1.42,1,0.58]
#L=[np.sqrt(0.5),np.sqrt(0.5),np.sqrt(2.0/3.0)]
L = [1, 1, 1]
#L=[0.4454,0.3143,0.2553]
print(np.sum(L))
# CA=[(-3.031883452592004, 3.031883452592004), (-1.619994146091692, -0.8276157453255935), (-1.6977602095460234, -0.7265946655975718)]
#CA = [(-0.7391244590957556, 0.7391244590957556), (-0.7422740927125862, 1.9756037937159996),
 #     (-2.11211741668124, 2.12020510030)]

CA=[(-0.5779611440942315, 0.5779611440942315), (-1.1887031063410372, 0.6453736884533061), (-1.7852473794934884, 1.8681718728028136)]
#CA = [(-18.2074 * np.pi / 180, 18.2074 * np.pi / 180), (-111.3415 * np.pi / 180, 111.3415 * np.pi / 180),(-111.3415 * np.pi / 180, 111.3415 * np.pi / 180)]

#CA = [(-42.35 * np.pi / 180, 42.35 * np.pi / 180), (-42.53 * np.pi / 180, 113.19 * np.pi / 180),(-121.02 * np.pi / 180, 121.48 * np.pi / 180)]
"""
CA = [(-30 * np.pi / 180, 30 * np.pi / 180), (-120 * np.pi / 180, 60 * np.pi / 180),
      (-130 * np.pi / 180, 130 * np.pi / 180)]
CA = [(-25 * np.pi / 180, 25 * np.pi / 180), (40 * np.pi / 180, 90 * np.pi / 180),
      (-60 * np.pi / 180, 120 * np.pi / 180)]
CA = [(-3.141592653589793, 3.141592653589793), (-0.9272951769138392, 2.2142957313467018), (1.8545903538276785, 1.8545903538276785)]
"""
# for a in range(len(CA)):
#    if CA[a][1] - CA[a][0] < step_size: CA[a] = (
#        CA[a][0] - terminate_threshold, CA[a][1] + terminate_threshold)
C_dot_A = CA.copy()
C_dot_A[0] = (-np.pi, np.pi)


def reliability_computation(r1, r2, r3):
    reliability_list = [r2 * r3, r1 * r3, r1 * r2, r2 * r3 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r2 * r3 - r1 * r2 * r3, r1 * r2 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3]
    conditional_reliability_list = []
    for p in reliability_list:
        conditional_reliability_list.append(
            np.power(((p - r1 * r2 * r3) / (r1 * r2 + r1 * r3 + r2 * r3 - 3 * r1 * r2 * r3)), 2))
    return conditional_reliability_list


cr_list = reliability_computation(r1, r2, r3)
print(cr_list)
indices = sorted_indices(cr_list)
print(indices)

def forward_kinematics_2R(theta, L,base_point):
    x1 = L[0] * np.cos(theta[0])
    y1 = L[0] * np.sin(theta[0])

    x2 = x1 + L[1] * np.cos(theta[0] + theta[1])
    y2 = y1 + L[1] * np.sin(theta[0] + theta[1])
    x2+=base_point[0]
    y2+=base_point[1]
    return np.array([x2, y2]).T.reshape((2, 1))

def forward_kinematics_3R(theta, L):
    """
    Computes the forward kinematics for a 3R planar robot arm.

    Parameters:
    theta1, theta2, theta3 : Joint angles in radians
    L : List containing the lengths of the three links

    Returns:
    (x, y) : Tuple containing the end-effector position (x, y)
    """
    theta = theta.flatten().tolist()
    x1 = L[0] * np.cos(theta[0])
    y1 = L[0] * np.sin(theta[0])

    x2 = x1 + L[1] * np.cos(theta[0] + theta[1])
    y2 = y1 + L[1] * np.sin(theta[0] + theta[1])

    x3 = x2 + L[2] * np.cos(theta[0] + theta[1] + theta[2])
    y3 = y2 + L[2] * np.sin(theta[0] + theta[1] + theta[2])

    return np.array([x3, y3]).T.reshape((2, 1))


def Jacobian_3R(theta, L):
    theta = theta.flatten().tolist()
    J = np.zeros((2, 3))
    J[0][0] = -L[0] * np.sin(theta[0]) - L[1] * np.sin(theta[0] + theta[1]) - L[2] * np.sin(
        theta[0] + theta[1] + theta[2])
    J[0][1] = - L[1] * np.sin(theta[0] + theta[1]) - L[2] * np.sin(theta[0] + theta[1] + theta[2])
    J[0][2] = - L[2] * np.sin(theta[0] + theta[1] + theta[2])
    J[1][0] = L[0] * np.cos(theta[0]) + L[1] * np.cos(theta[0] + theta[1]) + L[2] * np.cos(
        theta[0] + theta[1] + theta[2])
    J[1][1] = L[1] * np.cos(theta[0] + theta[1]) + L[2] * np.cos(theta[0] + theta[1] + theta[2])
    J[1][2] = L[2] * np.cos(theta[0] + theta[1] + theta[2])
    return J


def stepwise_ssm(theta, n_j, x_target, previous_n_j):
    a = np.array(n_j).reshape(-1)  # Converts 3x1 to 1D array
    b = np.array(previous_n_j).reshape(-1)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    if np.abs(angle_rad) > np.pi / 2:
        n_j = -n_j

    x = forward_kinematics_3R(theta, L)
    delta_x_step = x_target - x
    J_step = Jacobian_3R(theta, L)
    J_step_plus = np.linalg.pinv(J_step)
    corrected_delta_theta = np.dot(J_step_plus, delta_x_step)
    dq = n_j + corrected_delta_theta
    dq /= np.linalg.norm(dq)
    theta_next = theta + step_size * dq
    for i in range(len(theta_next)):
        theta_next[i] %= 2 * np.pi
        if theta_next[i] > np.pi: theta_next[i] -= 2 * np.pi
        if theta_next[i] < -np.pi: theta_next[i] += 2 * np.pi
    new_J = Jacobian_3R(theta_next, L)

    new_n_j = null_space(new_J)
    return theta_next, new_n_j, n_j


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
            return [[-np.pi,
                     np.pi]]
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


def find_intersection_points(ssm_theta_list):
    tof = False
    counter = 0
    ip_ranges = []
    isin_list = []
    for theta_index in range(len(ssm_theta_list)):
        theta = ssm_theta_list[theta_index]
        theta_flatten = theta.flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, C_dot_A)):
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
    return ip_ranges, tof


def find_critical_points(ssm_theta_list):
    thetas_cp_range = [[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]]
    thetas_cp_sum = [[], [], []]
    if len(ssm_theta_list) == 1:
        return ([ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
                , [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
                , [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]])
    if ssm_theta_list:
        thetas_cp_range[0] = [ssm_theta_list[0][0][0], ssm_theta_list[0][0][0]]
        thetas_cp_range[1] = [ssm_theta_list[0][1][0], ssm_theta_list[0][1][0]]
        thetas_cp_range[2] = [ssm_theta_list[0][2][0], ssm_theta_list[0][2][0]]
        previous_theta = ssm_theta_list[0]
        for theta_index in range(1, len(ssm_theta_list)):
            theta = ssm_theta_list[theta_index]
            for i in range(3):
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
    for j in range(3):
        thetas_cp_sum[j].append(thetas_cp_range[j])
        # thetas_cp_sum[j] = union_ranges(thetas_cp_sum[j])
    return thetas_cp_sum

def find_single_intersection(ssm_theta_list):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof

def dls(x_target, initial_config):
    q = np.array(initial_config).T.reshape((3, 1))
    step_num = 0
    error_threshold = 1e-6
    error = 10
    while error > error_threshold:
        x_current = forward_kinematics_3R(q, L)
        delta_x = x_target - x_current
        error = np.linalg.norm(x_target - x_current)
        J = Jacobian_3R(q, L)
        correction = J.T @ np.linalg.inv(J @ J.T + 0.5 ** 2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9: return []
        q = q + 0.1 * correction

        for i in range(len(q)):
            q[i] %= 2 * np.pi
            if q[i] > np.pi: q[i] -= 2 * np.pi
            if q[i] < -np.pi: q[i] += 2 * np.pi

        step_num += 1
        previous_x = x_current
        x_current = forward_kinematics_3R(q, L)
        consecutive_delta_x = np.linalg.norm(previous_x - x_current)
        if consecutive_delta_x < 1e-8:
            return []
    sol = q
    return sol

def ik_2r_single(x, y, L1, L2, base=(0.0, 0.0)):
    """
    Inverse kinematics for a 2R planar robot with arbitrary base point.
    Returns a single solution in radians.

    Args:
        x, y       : target end-effector coordinates (global frame)
        L1, L2     : link lengths
        prev_angle : previous known joint angle (radians)
        joint_index: 1 for joint1, 2 for joint2 (which joint is known)
        base       : (x0, y0) base coordinates of the robot in global frame

    Returns:
        (theta1, theta2) in radians
    """
    # Shift target into base-centered frame
    x0, y0 = base
    X = x - x0
    Y = y - y0

    r = np.hypot(X, Y)
    if r > L1 + L2:
        print('too far')
        raise ValueError("Target out of reach")
    if r < abs(L1 - L2):
        print('too close')
        raise ValueError("Target out of reach")

    # cos(theta2), clamp for safety
    c2 = (X*X + Y*Y - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = np.clip(c2, -1.0, 1.0)

    # elbow up and down
    s2_pos = np.sqrt(1.0 - c2*c2)
    s2_neg = -s2_pos

    def solve(s2):
        th2 = np.arctan2(s2, c2)
        th1 = np.arctan2(Y, X) - np.arctan2(L2*s2, L1 + L2*c2)
        return th1, th2

    sol_up = solve(s2_pos)
    sol_dn = solve(s2_neg)

    return sol_up, sol_dn

def lock_joint_1(theta_1):
    new_base_point=(L[0]*np.cos(theta_1),L[0]*np.sin(theta_1))
    return [L[1],L[2]],new_base_point

def lock_joint_2(theta_2):
    new_L1=np.sqrt(L[0]**2+L[1]**2-2.0*L[0]*L[1]*np.cos(np.pi-np.abs(theta_2)))
    return [new_L1,L[2]],(0,0)

def lock_joint_3(theta_3):
    new_L1=np.sqrt(L[1]**2+L[2]**2-2.0*L[1]*L[2]*np.cos(np.pi-np.abs(theta_3)))
    return [L[0],new_L1],(0,0)


def find_random_ssm(x_target, all_ssm_theta_list):
    ssm_found = False
    q = np.array(np.random.uniform(low=-np.pi, high=np.pi, size=(3,))).T.reshape((3, 1))
    step_num = 0
    error_threshold = 1e-6
    error = 10
    ik = False
    while error > error_threshold:
        x_current = forward_kinematics_3R(q, L)
        delta_x = x_target - x_current
        error = np.linalg.norm(x_target - x_current)
        J = Jacobian_3R(q, L)
        J_plus = np.linalg.pinv(J)
        # correction = np.dot(J_plus, delta_x)
        correction = J.T @ np.linalg.inv(J @ J.T + 0.5 ** 2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9: return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
        q = q + 0.1 * correction
        # correction=J.T @ np.linalg.inv(J @ J.T + 0.5**2 * np.eye(2)) @ delta_x
        # q = q + 0.1 * correction
        for i in range(len(q)):
            q[i] %= 2 * np.pi
            if q[i] > np.pi: q[i] -= 2 * np.pi
            if q[i] < -np.pi: q[i] += 2 * np.pi
        # iof, q = artificial_inverse_kinematics(q, x_target)
        # if not iof: return [], [[], [], [], []], all_ssm_theta_list, ssm_found
        step_num += 1
        # if step_num == 10000:
        #    print('IK failed or given position out of workspace')
        #    return [], [[], [], [], []], all_ssm_theta_list, ssm_found
        previous_x = x_current
        x_current = forward_kinematics_3R(q, L)
        consecutive_delta_x = np.linalg.norm(previous_x - x_current)
        if consecutive_delta_x < 1e-8:
            return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found,False
    ik = True
    sol = q

    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
    theta = sol
    theta_prime = theta.copy()

    J = Jacobian_3R(theta, L)
    n_j = null_space(J)
    old_n_j = n_j.copy()
    ssm_theta_list = [theta]
    num = 0
    threshold = 1
    lowest = step_size
    highest_delta_theta = -1000.0
    while True:
        if threshold <= terminate_threshold and num > 4: break
        old_theta = theta.copy()
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j)
        threshold = np.linalg.norm(theta - theta_prime)
        d_theta = np.linalg.norm(theta - old_theta)
        if threshold < lowest: lowest = threshold
        if d_theta > highest_delta_theta: highest_delta_theta = d_theta
        if num == 100 and highest_delta_theta > 2 * step_size:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    print('ssm guided to wrong direction.')
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
        n_j = new_n_j
        if num == 15000 and lowest > terminate_threshold:
            print('if you see this printed, then there are still spaces for efficiency improvement.')
            print('-------------------------------------------------------------------------------.')
            print('jump to another ssm for unknown reason, reassign starting point')
            theta_prime = theta
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm already exists.')
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found,False
            ssm_theta_list = [theta]
            num = -1
        num += 1
        ssm_theta_list.append(theta)
    all_ssm_theta_list.extend(ssm_theta_list)
    if x_target[0]==1.96875 and x_target[1]==0:
        list_of_lists = [arr.flatten().tolist() for arr in ssm_theta_list]  # Convert np.array to list
        with open('plot_list.txt', 'w') as file:
            json.dump(list_of_lists, file, indent=4)
    #print(f'found a new ssm with {num} points.')
    ssm_found = True

    #if x == -1.875 and y == 0.375:
    """
    if x == 2 and y == 0:
        ppoints = np.array(ssm_theta_list)

        figp = plt.figure()
        axp = figp.add_subplot(111, projection='3d')

        # Plot the points
        axp.scatter(ppoints[:, 0], ppoints[:, 1], ppoints[:, 2], c='b', marker='o')

        # Set labels
        axp.set_xlabel('theta1')
        axp.set_ylabel('theta2')
        axp.set_zlabel('theta3')

        # Set plot limits for better visualization
        axp.set_xlim([-np.pi, np.pi])
        axp.set_ylim([-np.pi, np.pi])
        axp.set_zlim([-np.pi, np.pi])

        plt.show()
    """


    #ip_ranges=[]
    ip_ranges, tof = find_intersection_points(ssm_theta_list)
    single_intersection_tf=find_single_intersection(ssm_theta_list)
    # ip_ranges = union_ranges(ip_ranges)
    cp_ranges = find_critical_points(ssm_theta_list)
    # print(min_theta1, max_theta1)
    return ik, ip_ranges, cp_ranges, all_ssm_theta_list, ssm_found, single_intersection_tf


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


def compute_beta_range(x, y):
    F_list = [False] * (8)
    target_x = np.array([x, y]).T.reshape((2, 1))
    all_smm_beta_range = []
    reliable_beta_ranges = [[], [], [], [], [], [], [],[]] # the last element for prefailure workspace(task space projection of artificial joint limits)
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []
    find_count = 0
    ssm_found = 0
    single_intersection_tf_all=False
    while find_count < ssm_finding_num and ssm_found < max_ssm:
        ik, iprs, cp_ranges, all_theta, ssm_found_tf, single_intersection_tf = find_random_ssm(
            target_x, all_theta)
        single_intersection_tf_all=single_intersection_tf_all or single_intersection_tf
        if not ik: break
        find_count += 1
        iprs = extend_ranges(iprs)

        for intersection_range in iprs:
            beta0_lm = CA[0][0] - intersection_range[1]
            beta0_um = CA[0][1] - intersection_range[0]
            #print(beta0_lm)
            #print(beta0_um)
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
            if cp_range[0] > -np.inf:
                theta1_ranges.append(cp_range)

        for cp_range in cp_ranges[1]:
            if cp_range[0] > -np.inf:
                theta2_ranges.append(cp_range)

        for cp_range in cp_ranges[2]:
            if cp_range[0] > -np.inf:
                theta3_ranges.append(cp_range)



    theta1_ranges_union = union_ranges(theta1_ranges)
    theta2_ranges_union = union_ranges(theta2_ranges)
    theta3_ranges_union = union_ranges(theta3_ranges)
    # print(theta1_ranges_union)
    # print(theta2_ranges_union)
    # print(theta3_ranges_union)
    ion1 = False
    min_beta1 = 0
    max_beta1 = 0
    theta1_ranges_union = extend_ranges(theta1_ranges_union)
    for tr in theta1_ranges_union:
        if CA[0][0] >= tr[0] and CA[0][1] <= tr[1]:
            ion1 = True
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
    if CA[1][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] <= has_negative_pi_range and CA[1][0] + 2 * np.pi >= has_positive_pi_range:
                ion2 = True
    if CA[1][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[1][1] - 2 * np.pi <= has_negative_pi_range and CA[1][0] >= has_positive_pi_range:
                ion2 = True
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
    if CA[2][0] < -np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] <= has_negative_pi_range and CA[2][0] + 2 * np.pi >= has_positive_pi_range:
                ion3 = True
    if CA[2][1] > np.pi:
        if has_negative_pi_range is not None and has_positive_pi_range is not None:
            if CA[2][1] - 2 * np.pi <= has_negative_pi_range and CA[2][0] >= has_positive_pi_range:
                ion3 = True
    min_beta3 = -np.pi
    max_beta3 = np.pi

    for index in range(len(beta0_ranges)):
        reliable_beta_ranges[-1].append(beta0_ranges[index])# prefailure workspace
        min_beta0, max_beta0 = beta0_ranges[index][0], beta0_ranges[index][1]
        # print(min_beta0, max_beta0)
        # print(min_beta1, max_beta1)
        # print(min_beta2, max_beta2)
        # print(min_beta3, max_beta3)
        if single_intersection_tf_all: F_list[-1] = True
        if ion1:
            if single_intersection_tf_all:F_list[0]=True
            min_beta_f_1 = max(min_beta0, min_beta1)
            max_beta_f_1 = min(max_beta0, max_beta1)
            if min_beta_f_1 <= max_beta_f_1:
                reliable_beta_ranges[0].append([min_beta_f_1, max_beta_f_1])
        if ion2:
            if single_intersection_tf_all:F_list[1] = True
            min_beta_f_2 = max(min_beta0, min_beta2)
            max_beta_f_2 = min(max_beta0, max_beta2)
            if min_beta_f_2 <= max_beta_f_2:

                reliable_beta_ranges[1].append([min_beta_f_2, max_beta_f_2])
        if ion3:
            if single_intersection_tf_all:F_list[2] = True
            min_beta_f_3 = max(min_beta0, min_beta3)
            max_beta_f_3 = min(max_beta0, max_beta3)
            if min_beta_f_3 <= max_beta_f_3:

                reliable_beta_ranges[2].append([min_beta_f_3, max_beta_f_3])
        if ion1 and ion2:
            if single_intersection_tf_all:F_list[3] = True
            min_beta_f_12 = max(min_beta0, min_beta1, min_beta2)
            max_beta_f_12 = min(max_beta0, max_beta1, max_beta2)
            if min_beta_f_12 <= max_beta_f_12:

                reliable_beta_ranges[3].append([min_beta_f_12, max_beta_f_12])
        if ion1 and ion3:
            if single_intersection_tf_all:F_list[4] = True
            min_beta_f_13 = max(min_beta0, min_beta1, min_beta3)
            max_beta_f_13 = min(max_beta0, max_beta1, max_beta3)
            if min_beta_f_13 <= max_beta_f_13:

                reliable_beta_ranges[4].append([min_beta_f_13, max_beta_f_13])
        if ion2 and ion3:
            if single_intersection_tf_all:F_list[5] = True
            min_beta_f_23 = max(min_beta0, min_beta2, min_beta3)
            max_beta_f_23 = min(max_beta0, max_beta2, max_beta3)
            if min_beta_f_23 <= max_beta_f_23:

                reliable_beta_ranges[5].append([min_beta_f_23, max_beta_f_23])
        if ion1 and ion2 and ion3:
            if single_intersection_tf_all:F_list[6] = True
            min_beta_f_ftw = max(min_beta0, min_beta1, min_beta2, min_beta3)
            max_beta_f_ftw = min(max_beta0, max_beta1, max_beta2, max_beta3)

            if min_beta_f_ftw <= max_beta_f_ftw:

                all_smm_beta_range.append([min_beta_f_ftw, max_beta_f_ftw])
                reliable_beta_ranges[6].append([min_beta_f_ftw, max_beta_f_ftw])

    for rbr_index in range(len(reliable_beta_ranges)):
        reliable_beta_ranges[rbr_index] = union_ranges(reliable_beta_ranges[rbr_index])
    return union_ranges(all_smm_beta_range), reliable_beta_ranges,F_list


def wedge_to_poly3d(wedge, z_value):
    theta1, theta2 = np.radians(wedge.theta1), np.radians(wedge.theta2)
    outer_radius = wedge.r
    inner_radius = wedge.r - wedge.width

    # Outer arc coordinates
    outer_arc_x = outer_radius * np.cos(np.linspace(theta1, theta2, 100))
    outer_arc_y = outer_radius * np.sin(np.linspace(theta1, theta2, 100))

    # Inner arc coordinates
    inner_arc_x = inner_radius * np.cos(np.linspace(theta2, theta1, 100))
    inner_arc_y = inner_radius * np.sin(np.linspace(theta2, theta1, 100))

    # Combine outer and inner arcs to form a closed polygon
    x = np.concatenate([outer_arc_x, inner_arc_x])
    y = np.concatenate([outer_arc_y, inner_arc_y])
    z = np.full_like(x, z_value)

    # Create a list of (x, y, z) tuples
    vertices = list(zip(x, y, z))
    return vertices

def main_function():
    num_reliable_ranges = 7
    cr_list.append(0)
    #color_list, sm = normalize_and_map_colors(cr_list)
    color_list, sm =normalize_and_map_greyscale(cr_list)
    final_wedges = []
    final_colors = []
    # z_levels = cr_list
    # np.linspace(-3, 3, num_reliable_ranges)

    #""" original approach
    section_length = 3.0 / sample_num
    x_values = (np.arange(sample_num) + 0.5) * section_length
    y_values = np.zeros(sample_num)
    points = np.column_stack((x_values, y_values))
    #"""

    """ uniform sample
    d = 3.0 / sample_num
    edges = np.arange(-3, 3 + 1e-6, d)
    n_cells = edges.size - 1             # cells per axis
    centers = np.linspace(-3 + d/2, 3 - d/2, n_cells)
    x_c, y_c = np.meshgrid(centers, centers)
    mask = x_c**2 + y_c**2 <= 9
    x_values=x_c[mask]
    y_values=y_c[mask]
    points = np.column_stack((x_values, y_values))
    print(len(points))
    """

    """ uniform and random
    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(6, 6))

    # circle boundary
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(3*np.cos(theta), 3*np.sin(theta), linewidth=1.2, color='black')
    circle = patches.Circle((0, 0), np.sum(L), edgecolor=color_list[-1], facecolor=color_list[-1], linewidth=0, zorder=0)
    ax.add_patch(circle)
    """





    """ random sample
    N = 812
    theta = np.random.rand(N) * 2 * np.pi
    r = np.sqrt(np.random.rand(N)) * 3
    x_values = r * np.cos(theta )
    y_values = r * np.sin(theta )
    points = np.column_stack((x_values, y_values))
    diam = 3.0 / sample_num
    """

    #""" original approach
    ring_width = 2.0*x_values[0]
    
    
    fig3, ax2d3 = plt.subplots(figsize=(6, 6))
    
    # Set limits for the 2D plot

    ax2d3.set_xlim(-np.sum(L), np.sum(L))
    ax2d3.set_ylim(-np.sum(L),np.sum(L))
    ax2d3.set_aspect('equal')
    ax2d3.set_xlabel("x", fontsize=25)
    ax2d3.set_ylabel("y", fontsize=25)
    tick_positions = [-2,-1, 0, 1,2]
    ax2d3.set_xticks(tick_positions)
    ax2d3.set_yticks(tick_positions)
    circle = patches.Circle((0, 0), np.sum(L), edgecolor=color_list[-1], facecolor=color_list[-1], linewidth=0, zorder=0)
    ax2d3.add_patch(circle)
    ax2d3.axhline(0, color='black', linewidth=1)  # y=0
    ax2d3.axvline(0, color='black', linewidth=1)  # x=0
    x_ticks = np.linspace(0, 3, sample_num + 1)
    tick_length = 0.05
    for x in x_ticks:
        ax2d3.plot([x, x], [0, tick_length], color='black', linewidth=1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N=0
    #"""

    def plot_prefailure_boundary(ax,sample_num):
        section_length = 3.0 / sample_num
        local_x_values = (np.arange(sample_num) + 0.5) * section_length
        local_y_values = np.zeros(sample_num)
        points = np.column_stack((local_x_values, local_y_values))
        ring_width = 2.0 * local_x_values[0]
        polys2d=[]
        for i in range(len(points)):
            point = points[i]
            x, y = point
            beta_ranges, reliable_beta_ranges, F_list = compute_beta_range(x, y)
            prefailure_b_r = reliable_beta_ranges[-1]
            for beta_range in prefailure_b_r:
                theta1 = np.degrees(beta_range[0])  # Start angle (-π)
                theta2 = np.degrees(beta_range[1])  # End angle (π)

                # Calculate the outer radius for the ring (x + ring_width / 2)
                outer_radius = x + ring_width / 2.0
                inner_radius = x - ring_width / 2.0
                polys2d.append(
                    wedge_polygon((0, 0), outer_radius, inner_radius, theta1, theta2, n=64)
                )
        if polys2d:  # guard against empty list
            shape2d = unary_union(polys2d)  # remove internal boundaries
            plot_exterior_boundary(ax, shape2d,  # change to ax for uniform and random, ax2d3 for original
                                   color='k', linewidth=1.8)


    start = time.perf_counter()
    for i in range(len(points)):
        point = points[i]
        x, y = point

        print(point)
        beta_ranges, reliable_beta_ranges,F_list = compute_beta_range(x, y)  # Get multiple beta ranges

        # for b_r_index in range(len(reliable_beta_ranges)):


        """ orignial
        prefailure_b_r = reliable_beta_ranges[-1]
        for beta_range in prefailure_b_r:
            theta1 = np.degrees(beta_range[0])  # Start angle (-π)
            theta2 = np.degrees(beta_range[1])  # End angle (π)
    
            # Calculate the outer radius for the ring (x + ring_width / 2)
            outer_radius = x + ring_width / 2.0
            inner_radius = x - ring_width / 2.0
            polys2d.append(
                wedge_polygon((0, 0), outer_radius, inner_radius, theta1, theta2, n=64)
            )
        """


        for b_r_index in indices:
            b_r = reliable_beta_ranges[b_r_index]
            # z_level = z_levels[b_r_index]
            z_level = b_r_index * 2
            color = color_list[b_r_index]


            f_to=F_list[b_r_index]
            if f_to:
                """ uniform sample
                rect = Rectangle(
                    (x - d / 2, y - d / 2),
                    d, d,
                    facecolor=color,
                    edgecolor='none',
                    zorder=b_r_index + 1,
                    linewidth=0,
                    alpha=1.0
                )
                ax.add_patch(rect)
                """

                """random sample
                circ = Circle(
                    (x, y),  # center
                    radius=diam / 2,  # radius = diameter / 2
                    facecolor=color,
                    edgecolor='none',
                    zorder=b_r_index + 1,
                    linewidth=0,
                    alpha=1.0
                )
                ax.add_patch(circ)
                """

            for beta_range in b_r:
                #""" original approach
                # Compute angles in degrees (as required by Wedge)
                theta1 = np.degrees(beta_range[0])  # Start angle (-π)
                theta2 = np.degrees(beta_range[1])  # End angle (π)
    
                # Calculate the outer radius for the ring (x + ring_width / 2)
                outer_radius = x + ring_width / 2.0
                inner_radius = x-ring_width/2.0
                if b_r_index == 6: N += np.pi * (outer_radius ** 2 -inner_radius**2)/(2*np.pi)*(beta_range[1]-beta_range[0])
    
                wedge = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1, theta2=theta2,
                    width=ring_width,
                    facecolor=color,  # Set face color to match 3D plot
                    edgecolor=color,
                    alpha=1.0,
                    zorder=b_r_index+1
                )
    
                wedge_2d = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1, theta2=theta2,
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0,
                    zorder=b_r_index+1
                )
                
                poly3d = wedge_to_poly3d(wedge, z_level)
                # Add the wedge to the plot
                # ax.add_patch(wedge)
                ax.add_collection3d(
                    Poly3DCollection(
                        [poly3d],
                        facecolor=color,
                        edgecolor=color,
                        alpha=1.0
                    )
                )
    
                ax2d3.add_patch(wedge_2d)  # Add wedge to 2D plot
    
                if b_r_index == len(reliable_beta_ranges) - 1:
                    final_wedges.append(wedge)
                    final_colors.append(color)
    
                #"""

    """ original
    if polys2d:  
        shape2d = unary_union(polys2d)  # remove internal boundaries
        plot_exterior_boundary(ax, shape2d,  # change to ax for uniform and random, ax2d3 for original
                                   color='k', linewidth=1.8)
    """
    #plot_prefailure_boundary(ax,sample_num)


    """ uniform sample
    # draw grid lines
    for e in edges:
        ax.plot([edges[0], edges[-1]], [e, e], linewidth=1, alpha=1,zorder=8)  # horizontal
        ax.plot([e, e], [edges[0], edges[-1]], linewidth=1, alpha=1,zorder=8)  # vertical

    # plot kept centers in black
    ax.scatter(points[:, 0], points[:, 1], s=8, color='black', zorder=8)
    #"""
    end = time.perf_counter()
    print(f"Loop took {end - start:.6f} seconds")
    """ uniform and random

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x", fontsize=25)
    ax.set_ylabel("y", fontsize=25)
    tick_positions = [-2,-1, 0, 1,2]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax.tick_params(axis='y', labelsize=18)  # Increase font size for Y-axis ticks
    plt.tight_layout()
    plt.show()
    """


    #""" original approach
    #ax2d3.scatter(-1.875,0.375, s=8, color='black')
    ax2d3.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax2d3.tick_params(axis='y', labelsize=18)  # Increase font size for Y-axis ticks
    # cbar = plt.colorbar(sm, ax=ax2d3, label='Reliability Spectrum')  # Ensure colorbar is linked to the mappable
    #ax2d3.set_title("Fault tolerant workspace")
    fig3.show()
    
    # Set the plot limits to range from -3 to 3 for both x and y axes
    radius=np.sum(L)
    ax.set_xlim(-np.sum(L), np.sum(L))
    ax.set_ylim(-np.sum(L), np.sum(L))
    ax.set_zlim(0, 12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    tick_positions = [-2, 0, 2]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    # ax.tick_params(axis='z', labelsize=14)
    
    # ax.set_zlabel("Failable Joints")
    
    # Set aspect ratio to be equal for correct visualization of circles
    ax.set_aspect('equal')
    print(N)
    
    # Add title and display the plot
    # plt.title("Planar3R Reliable work spaces")
    cbar = plt.colorbar(sm, ax=ax)  # Ensure colorbar is linked to the mappable
    tick_positions = np.arange(0.1, 1.1, 0.1)  # 1.1 ensures 1.0 is included
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in tick_positions])
    # Define the new labels and corresponding tick positionsq
    z_tick_labels = [r'$\mathit{F}=\{1\}$', r'$\mathit{F}=\{2\}$', r'$\mathit{F}=\{3\}$',
                     r'$\mathit{F}=\{1,2\}$', r'$\mathit{F}=\{1,3\}$', r'$\mathit{F}=\{2,3\}$', r'$\mathit{F}=\{1,2,3\}$']
    ax.tick_params(axis='z', labelsize=18) 
    z_tick_positions = [0, 2, 4, 6, 8, 10, 12]  # These match your z_level values
    
    # Set the new labels on the Z-axis
    ax.set_zticks(z_tick_positions)
    ax.set_zticklabels(z_tick_labels)
    fig2, ax2d = plt.subplots(figsize=(6, 6))
    
    for wedge, color in zip(final_wedges, final_colors):
        ax2d.add_patch(wedge)  # Add the stored wedge
        wedge.set_facecolor(color)  # Ensure colors match
        wedge.set_edgecolor(color)
    
    # Configure the 2D plot
    ax2d.set_xlim(-3, 3)
    ax2d.set_ylim(-3, 3)
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    ax2d.set_aspect('equal')
    ax2d.set_xlabel("x", fontsize=25)
    ax2d.set_ylabel("y", fontsize=25)
    ax2d.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
    ax2d.tick_params(axis='y', labelsize=18)  # Increase font size for Y-axis ticks
    #ax2d.set_title("Fault tolerant workspace")
    fig2.show()
    plt.show()
    #"""

def fold_offset(La, Lb, theta_locked):
    # orientation offset of the rigid pair relative to the first link
    return np.arctan2(Lb*np.sin(theta_locked), La + Lb*np.cos(theta_locked))

main_function()

"""
import json
results = []

p1 = (0.6015959713, 0.2279385242)#y+=0.4
p2 = (0.7384040287, -0.1479385242)
#p1 = (0.4015959713, -0.4279385242)#y+=0.4
#p2 = (0.5384040287, -0.7479385242)

sample_number=40
points = sample_line(p1, p2, sample_number)
#random_lock_time = np.random.randint(1, sample_number+2)
#print(random_lock_time)
random_lock_time= 35 #7,14,21,28,35,
random_lock_joint = np.random.randint(0, 3)
random_lock_joint=2
#q = np.array([10*np.pi/180.0,10*np.pi/180.0,10*np.pi/180.0]).T.reshape((3, 1))
#print(forward_kinematics_3R(q, L))



initial_config = [np.random.uniform(low, high) for (low, high) in CA]
base=(0,0)
new_L=[]
reference_index=0
true_index=0
reference_angle=0
locked_angle=0
success=0
for index, tp in enumerate(points) :
    print(tp)
    (x, y)=tp
    if index==random_lock_time:
        match random_lock_joint:
            case 0:
                locked_angle=initial_config[0]
                new_L,new_base=lock_joint_1(locked_angle)
                base =new_base
                reference_index=1
                true_index=0
                reference_angle=initial_config[1]
            case 1:
                locked_angle = initial_config[1]
                new_L,new_base=lock_joint_2(locked_angle)
                base = new_base
                reference_index = 2
                true_index=1
                reference_angle = initial_config[2]
            case 2:
                locked_angle=initial_config[2]
                new_L,new_base=lock_joint_3(locked_angle)
                base = new_base
                reference_index = 1
                true_index=2
                reference_angle = initial_config[0]
    if index>=random_lock_time:
        print("-----------------------")
        print(f"joint {true_index} is locked at angle {locked_angle / np.pi * 180.0}")
        print("-----------------------")
        try:
            sol1, sol2 = ik_2r_single(
                x, y,
                new_L[0], new_L[1],
                base=base
            )
            #x_t,y_t=forward_kinematics_2R([theta1, theta2],new_L,base)
            #print(x_t)
            #print(y_t)
        except ValueError as e:
            print(f"IK failed at index {index}: {e}")
            break  # quit the loop
        match random_lock_joint:
            case 0:
                 theta10,theta20=sol1
                 theta11,theta21=sol2
                 test_config1=[locked_angle,theta10-locked_angle,theta20]
                 test_config2 = [locked_angle, theta11 - locked_angle, theta21]
                 previous=initial_config
                 initial_config = test_config2
                 if math.sqrt(sum((a - b) ** 2 for a, b in zip(test_config1, previous)))<math.sqrt(sum((a - b) ** 2 for a, b in zip(test_config2, previous))):
                     initial_config=test_config1
                 print(initial_config[0] / np.pi * 180.0)
                 print(initial_config[1] / np.pi * 180.0)
                 print(initial_config[2] / np.pi * 180.0)
                 joint1 = float(initial_config[0])
                 joint4 = float(initial_config[1])
                 joint6 = float(initial_config[2])

                 # Append full 7-joint configuration
                 full_config = [
                     joint1,  # joint 1
                     90.0 * np.pi / 180.0,  # joint 2 fixed at 90 deg
                     90.0 * np.pi / 180.0,  # joint 3 fixed at 90 deg
                     joint4,  # joint 4
                     0.0,  # joint 5 fixed at 0 deg
                     joint6,  # joint 6
                     180.0 * (np.pi-1e-3) / 180.0  # joint 7 fixed at 180 deg
                 ]
                 results.append(full_config)
                 #print(f'computed 2R position:{forward_kinematics_2R([theta1, theta2],new_L, base)}')
                 #print(f'computed 3R position:{forward_kinematics_3R(np.array(initial_config), L)}')
                 #print(f'true position:{tp}')
            case 1:
                 phi2 = fold_offset(L[0], L[1], locked_angle)
                 theta10, theta20 = sol1
                 theta11, theta21 = sol2
                 test_config1 = [theta10 - phi2, locked_angle, theta20 - locked_angle + phi2]
                 test_config2 = [theta11 - phi2, locked_angle, theta21 - locked_angle + phi2]
                 previous = initial_config
                 initial_config = test_config2
                 if math.sqrt(sum((a - b) ** 2 for a, b in zip(test_config1, previous))) < math.sqrt(
                         sum((a - b) ** 2 for a, b in zip(test_config2, previous))):
                     initial_config = test_config1

                 print(initial_config[0]/ np.pi * 180.0)
                 print(initial_config[1] / np.pi * 180.0)
                 print(initial_config[2] / np.pi * 180.0)
                 joint1 = float(initial_config[0])
                 joint4 = float(initial_config[1])
                 joint6 = float(initial_config[2])

                 # Append full 7-joint configuration
                 full_config = [
                     joint1,  # joint 1
                     90.0 * np.pi / 180.0,  # joint 2 fixed at 90 deg
                     90.0 * np.pi / 180.0,  # joint 3 fixed at 90 deg
                     joint4,  # joint 4
                     0.0,  # joint 5 fixed at 0 deg
                     joint6,  # joint 6
                     180.0 * (np.pi-1e-3)/ 180.0  # joint 7 fixed at 180 deg
                 ]
                 results.append(full_config)
                 #print(f'computed 2R position:{forward_kinematics_2R([theta1, theta2], new_L, base)}')
                 #print(f'computed 3R position:{forward_kinematics_3R(np.array(initial_config), L)}')
                 #print(f'true position:{tp}')
            case 2:
                 phi3 = fold_offset(L[1], L[2], locked_angle)
                 theta10, theta20 = sol1
                 theta11, theta21 = sol2
                 test_config1 = [theta10, theta20 - phi3, locked_angle]
                 test_config2 = [theta11, theta21 - phi3, locked_angle]
                 previous = initial_config
                 initial_config = test_config2
                 if math.sqrt(sum((a - b) ** 2 for a, b in zip(test_config1, previous))) < math.sqrt(
                         sum((a - b) ** 2 for a, b in zip(test_config2, previous))):
                     initial_config = test_config1

                 print(initial_config[0] / np.pi * 180.0)
                 print(initial_config[1] / np.pi * 180.0)
                 print(initial_config[2] / np.pi * 180.0)
                 joint1 = float(initial_config[0])
                 joint4 = float(initial_config[1])
                 joint6 = float(initial_config[2])

                 # Append full 7-joint configuration
                 full_config = [
                     joint1,  # joint 1
                     90.0 * np.pi / 180.0,  # joint 2 fixed at 90 deg
                     90.0 * np.pi / 180.0,  # joint 3 fixed at 90 deg
                     joint4,  # joint 4
                     0.0,  # joint 5 fixed at 0 deg
                     joint6,  # joint 6
                     180.0 * (np.pi-1e-3) / 180.0  # joint 7 fixed at 180 deg
                 ]
                 results.append(full_config)
                 #print(f'computed 2R position:{forward_kinematics_2R([theta1, theta2], new_L, base)}')
                 #print(f'computed 3R position:{forward_kinematics_3R(np.array(initial_config), L)}')
                 #print(f'true position:{tp}')
        print("-----------------------")
        success+=1
    else:
        sol = dls(np.array([x, y]).reshape((2, 1)), initial_config)
        while not all(low <= s <= high for s, (low, high) in zip(sol, CA)) or sol == []:
            initial_config = [np.random.uniform(low, high) for (low, high) in CA]
            sol = dls(np.array([x, y]).reshape((2, 1)), initial_config)
        initial_config=sol
        joint1 = float(initial_config[0])
        joint4 = float(initial_config[1])
        joint6 = float(initial_config[2])

        # Append full 7-joint configuration
        full_config = [
            joint1,  # joint 1
            90.0 * np.pi / 180.0,  # joint 2 fixed at 90 deg
            90.0 * np.pi / 180.0,  # joint 3 fixed at 90 deg
            joint4,  # joint 4
            0.0,  # joint 5 fixed at 0 deg
            joint6,  # joint 6
            180.0 * (np.pi-1e-3) / 180.0  # joint 7 fixed at 180 deg
        ]
        results.append(full_config)
        #if not all(low <= initial_config <= high for initial_config, (low, high) in zip(initial_config, CA)):print("At least one angle is out of range")
        for i in sol:
            print(i/np.pi*180.0)
        #print(sol)
        print("-----------------------")
        success+=1
if success == 42: print('task complete!')

DT = 0.5  # seconds between trajectory points

traj = {
    "joint_names": ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"],
    "points": []
}

for i, q in enumerate(results):
    sec = int(i * DT)
    nanosec = int((i * DT - sec) * 1e9)
    traj["points"].append({
        "positions": [float(x) for x in q],
        "time_from_start": {"sec": sec, "nanosec": nanosec}
    })

# Compact JSON payload
payload = json.dumps(traj, separators=(",", ":"))

# Compose the final ROS2 command (single-line, paste-ready)
ros_cmd = f"ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory '{payload}'"

# Save it to a file
with open("ros2_trajectory_command.txt", "w") as f:
    f.write(ros_cmd)

print(f"✅ Saved ready-to-run ROS2 command to ros2_trajectory_command.txt")
print("➡  Open it and paste the entire command into your terminal to execute.")

"""
