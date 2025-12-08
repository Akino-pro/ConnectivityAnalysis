import json
import math
import time

import cv2
import numpy as np
from matplotlib.patches import Wedge
from scipy.linalg import null_space
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from greyscale_experiment import compute_connectivity
from helper_functions import  sorted_indices, normalize_and_map_greyscale

step_size = 0.01
# terminate_threshold = step_size / 2.0
terminate_threshold = 9.0 / 5.0 * step_size
ssm_finding_num = 20
max_ssm = 2
sample_num = 64
# r1 =0.3
# r2 =0.2
# r3 =0.1
r1 = 1.0 / 3.0
r2 = 1.0 / 3.0
r3 = 1.0 / 3.0
section_length = 3.0 / sample_num
x_values = (np.arange(sample_num) + 0.5) * section_length
y_values = np.zeros(sample_num)
points = np.column_stack((x_values, y_values))
ring_width = 2.0 * x_values[0]





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
#print(cr_list)
cr_list=[math.pow(cr,2) for cr in cr_list]
#print(cr_list)
#cr_list=[0.11, 0.11, 0.11, 0.44, 0.44,0.44, 0.44]
indices = sorted_indices(cr_list)
cr_list.append(0)
color_list, sm = normalize_and_map_greyscale(cr_list)



def forward_kinematics_2R(theta, L, base_point):
    x1 = L[0] * np.cos(theta[0])
    y1 = L[0] * np.sin(theta[0])

    x2 = x1 + L[1] * np.cos(theta[0] + theta[1])
    y2 = y1 + L[1] * np.sin(theta[0] + theta[1])
    x2 += base_point[0]
    y2 += base_point[1]
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


def stepwise_ssm(theta, n_j, x_target, previous_n_j,L):
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


def find_intersection_points(ssm_theta_list,C_dot_A):
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


def find_single_intersection(ssm_theta_list,CA):
    tof = False
    for theta_index in range(len(ssm_theta_list)):
        theta_flatten = ssm_theta_list[theta_index].flatten()
        if all(r[0] <= v <= r[1] for v, r in zip(theta_flatten, CA)):
            tof = True

    return tof




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
    c2 = (X * X + Y * Y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)

    # elbow up and down
    s2_pos = np.sqrt(1.0 - c2 * c2)
    s2_neg = -s2_pos

    def solve(s2):
        th2 = np.arctan2(s2, c2)
        th1 = np.arctan2(Y, X) - np.arctan2(L2 * s2, L1 + L2 * c2)
        return th1, th2

    sol_up = solve(s2_pos)
    sol_dn = solve(s2_neg)

    return sol_up, sol_dn




def find_random_ssm(x_target, all_ssm_theta_list,L,CA):
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
        correction = J.T @ np.linalg.inv(J @ J.T + 0.5 ** 2 * np.eye(2)) @ delta_x
        if np.linalg.norm(correction) <= 1e-9: return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found, False
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
            return ik, [], [[], [], [], []], all_ssm_theta_list, ssm_found, False
    ik = True
    sol = q

    for configuration in all_ssm_theta_list:
        if np.linalg.norm(configuration - sol) <= terminate_threshold:
            # print('ssm already exists.')
            return ik, [], [[], [], []], all_ssm_theta_list, ssm_found, False
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
        theta, new_n_j, old_n_j = stepwise_ssm(theta, n_j, x_target, old_n_j,L)
        threshold = np.linalg.norm(theta - theta_prime)
        d_theta = np.linalg.norm(theta - old_theta)
        if threshold < lowest: lowest = threshold
        if d_theta > highest_delta_theta: highest_delta_theta = d_theta
        if num == 100 and highest_delta_theta > 2 * step_size:
            # check if the searching has been guided to an searched smm in 100 steps.
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    print('ssm guided to wrong direction.')
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found, False
        n_j = new_n_j
        if num == 15000 and lowest > terminate_threshold:
            print('if you see this printed, then there are still spaces for efficiency improvement.')
            print('-------------------------------------------------------------------------------.')
            print('jump to another ssm for unknown reason, reassign starting point')
            theta_prime = theta
            for configuration in all_ssm_theta_list:
                if np.linalg.norm(configuration - theta) <= terminate_threshold:
                    # print('ssm already exists.')
                    return ik, [], [[], [], []], all_ssm_theta_list, ssm_found, False
            ssm_theta_list = [theta]
            num = -1
        num += 1
        ssm_theta_list.append(theta)
    all_ssm_theta_list.extend(ssm_theta_list)
    #print(f'found a new ssm with {num} points.')
    ssm_found = True

    C_dot_A = CA.copy()
    C_dot_A[0] = (-np.pi, np.pi)
    ip_ranges, tof = find_intersection_points(ssm_theta_list,C_dot_A)
    single_intersection_tf = find_single_intersection(ssm_theta_list,CA)
    cp_ranges = find_critical_points(ssm_theta_list)
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


def compute_beta_range(x, y, L, CA):
    """
    Memory-safe, correctness-preserved version.
    All mathematical logic remains unchanged.
    """

    target_x = np.array([x, y]).reshape((2, 1))

    all_smm_beta_range = []
    reliable_beta_ranges = [[] for _ in range(8)]

    # temporary containers
    all_theta = []
    beta0_ranges = []
    theta1_ranges = []
    theta2_ranges = []
    theta3_ranges = []

    find_count = 0
    ssm_found = 0
    single_intersection_tf_all = False

    # ---------------------------------------------------
    # small helpers (moved outside loops to avoid rebind)
    # ---------------------------------------------------

    def normalize_beta0_range(b0, b1):
        """Normalize range into [-pi,pi] segments."""
        if b1 - b0 >= 2 * np.pi:
            return [[-np.pi, np.pi]]
        elif b0 < -np.pi:
            return [[b0 + 2*np.pi, np.pi], [-np.pi, b1]]
        elif b1 > np.pi:
            return [[-np.pi, b1 - 2*np.pi], [b0, np.pi]]
        return [[b0, b1]]

    def check_CA_on_theta_union(union_list, CA_k):
        ak, bk = CA_k
        ion = False
        neg_pi_r = None
        pos_pi_r = None

        for t0, t1 in union_list:
            if t0 == -np.pi:
                neg_pi_r = t1
            if t1 == np.pi:
                pos_pi_r = t0
            if ak >= t0 and bk <= t1:
                ion = True

        # wrap cases
        if ak < -np.pi and neg_pi_r is not None and pos_pi_r is not None:
            if (bk <= neg_pi_r) and (ak + 2*np.pi >= pos_pi_r):
                ion = True

        if bk > np.pi and neg_pi_r is not None and pos_pi_r is not None:
            if (bk - 2*np.pi <= neg_pi_r) and (ak >= pos_pi_r):
                ion = True

        return ion

    # ---------------------------------------------------
    #        MAIN SSM SEARCH LOOP
    # ---------------------------------------------------
    while find_count < ssm_finding_num and ssm_found < max_ssm:

        ik, iprs, cp_ranges, all_theta, ssm_found_tf, single_tf = find_random_ssm(
            target_x, all_theta, L, CA
        )

        single_intersection_tf_all |= single_tf

        if not ik:
            break

        find_count += 1
        iprs = extend_ranges(iprs)

        # build beta0 intervals
        CA0_low, CA0_high = CA[0]
        for (lm, um) in iprs:
            b0_low = CA0_low - um
            b0_high = CA0_high - lm
            for r in normalize_beta0_range(b0_low, b0_high):
                beta0_ranges.append(r)

        if ssm_found_tf:
            ssm_found += 1
            find_count = 0

        # collect theta ranges (avoid infinite)
        for cp in cp_ranges[0]:
            if cp[0] > -np.inf:
                theta1_ranges.append(cp)

        for cp in cp_ranges[1]:
            if cp[0] > -np.inf:
                theta2_ranges.append(cp)

        for cp in cp_ranges[2]:
            if cp[0] > -np.inf:
                theta3_ranges.append(cp)

    # ---------------------------------------------------
    # union theta ranges
    # ---------------------------------------------------
    theta1_union = extend_ranges(union_ranges(theta1_ranges))
    theta2_union = union_ranges(theta2_ranges)
    theta3_union = union_ranges(theta3_ranges)

    # ---------------------------------------------------
    # determine ion flags and β ranges
    # ---------------------------------------------------
    ion1 = False
    min_beta1 = 0
    max_beta1 = 0

    CA1_low, CA1_high = CA[0]

    for tr in theta1_union:
        t0, t1 = tr
        if CA1_low >= t0 and CA1_high <= t1:
            ion1 = True
            min_beta1 = CA1_high - t1
            max_beta1 = CA1_low - t0
            if (max_beta1 - min_beta1 >= 2*np.pi) or (t0 == -np.pi and t1 == np.pi):
                min_beta1, max_beta1 = -np.pi, np.pi

    ion2 = check_CA_on_theta_union(theta2_union, CA[1])
    ion3 = check_CA_on_theta_union(theta3_union, CA[2])

    min_beta2 = -np.pi
    max_beta2 = np.pi
    min_beta3 = -np.pi
    max_beta3 = np.pi

    # ---------------------------------------------------
    # combine beta0 with availability flags
    # ---------------------------------------------------
    for (min_b0, max_b0) in beta0_ranges:

        # always add prefailure
        reliable_beta_ranges[-1].append([min_b0, max_b0])

        # J1 only
        if ion1:
            lo = max(min_b0, min_beta1)
            hi = min(max_b0, max_beta1)
            if lo <= hi:
                reliable_beta_ranges[0].append([lo, hi])

        # J2
        if ion2:
            lo = max(min_b0, min_beta2)
            hi = min(max_b0, max_beta2)
            if lo <= hi:
                reliable_beta_ranges[1].append([lo, hi])

        # J3
        if ion3:
            lo = max(min_b0, min_beta3)
            hi = min(max_b0, max_beta3)
            if lo <= hi:
                reliable_beta_ranges[2].append([lo, hi])

        # J1 & J2
        if ion1 and ion2:
            lo = max(min_b0, min_beta1, min_beta2)
            hi = min(max_b0, max_beta1, max_beta2)
            if lo <= hi:
                reliable_beta_ranges[3].append([lo, hi])

        # J1 & J3
        if ion1 and ion3:
            lo = max(min_b0, min_beta1, min_beta3)
            hi = min(max_b0, max_beta1, max_beta3)
            if lo <= hi:
                reliable_beta_ranges[4].append([lo, hi])

        # J2 & J3
        if ion2 and ion3:
            lo = max(min_b0, min_beta2, min_beta3)
            hi = min(max_b0, max_beta2, max_beta3)
            if lo <= hi:
                reliable_beta_ranges[5].append([lo, hi])

        # J1 & J2 & J3 (FTW)
        if ion1 and ion2 and ion3:
            lo = max(min_b0, min_beta1, min_beta2, min_beta3)
            hi = min(max_b0, max_beta1, max_beta2, max_beta3)
            if lo <= hi:
                interval = [lo, hi]
                all_smm_beta_range.append(interval)
                reliable_beta_ranges[6].append(interval)

    # ---------------------------------------------------
    # Final union on each entry
    # ---------------------------------------------------
    for i in range(8):
        reliable_beta_ranges[i] = union_ranges(reliable_beta_ranges[i])

    # SAFE cleanup of giant temporary lists
    del theta1_ranges[:]
    del theta2_ranges[:]
    del theta3_ranges[:]
    del beta0_ranges[:]

    return union_ranges(all_smm_beta_range), reliable_beta_ranges



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

def figure_to_gray_image(fig):
    """
    Render a Matplotlib figure to a grayscale image (float32, 0–255).
    Uses buffer_rgba() so we don't have to manually reshape.
    """
    # Draw the figure into the canvas
    fig.canvas.draw()

    # Get an RGBA array directly: shape (H, W, 4)
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (h, w, 4)

    # Convert RGBA -> RGB
    img_rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)

    # Convert RGB -> grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_gray.astype(np.float32)


"""
def planar_3R_greyscale_connectivity_analysis(L, CA):
    

    #print(L)
    #print(CA)
    fig3, ax2d3 = plt.subplots(figsize=(6, 6))

    ax2d3.set_xlim(-np.sum(L), np.sum(L))
    ax2d3.set_ylim(-np.sum(L), np.sum(L))
    ax2d3.set_aspect('equal')
    ax2d3.axis('off')

    #start = time.perf_counter()
    total_area = 0.0

    for i in range(len(points)):

        point = points[i]
        x, y = point

        beta_ranges, reliable_beta_ranges = compute_beta_range(x, y, L, CA)

        ranges_to_calculate_area = []
        for b_r_index in indices:
            b_r = reliable_beta_ranges[b_r_index]
            color = color_list[b_r_index]
            for beta_range in b_r:
                ranges_to_calculate_area.append(beta_range)
                theta1 = np.degrees(beta_range[0])  # Start angle
                theta2 = np.degrees(beta_range[1])  # End angle

                outer_radius = x + ring_width / 2.0

                wedge_2d = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1, theta2=theta2,
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0,
                    zorder=b_r_index + 1
                )

                ax2d3.add_patch(wedge_2d)

        if ranges_to_calculate_area:
            merged_ranges = union_ranges(ranges_to_calculate_area)

            inner_radius = x - ring_width / 2.0
            outer_radius = x + ring_width / 2.0
            ring_area = np.pi * (outer_radius ** 2 - inner_radius ** 2)  # full ring

            for beta_range in merged_ranges:
                beta_start, beta_end = beta_range
                angular_fraction = (beta_end - beta_start) / (2 * np.pi)
                total_area += ring_area * angular_fraction


    #end = time.perf_counter()
    #print(f"Loop took {end - start:.6f} seconds")

    # From here on, we assume we are within time budget
    img_gray = figure_to_gray_image(fig3)

    #connectivity_value, depths, conn_curve = compute_connectivity(img_gray)
    connectivity_value=compute_connectivity(img_gray)
    plt.close(fig3)

    #print(f"area: {total_area:.6f}")
    #print(f"Final connectivity: {connectivity_value:.6f}")
    #print(connectivity_value)
    #print(total_area * connectivity_value)

    #return total_area * connectivity_value
    return connectivity_value
"""

import gc
import matplotlib
matplotlib.use("Agg")   # disable GUI backend to reduce RAM
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def planar_3R_greyscale_connectivity_analysis(L, CA):

    # KEEP 5x5 FOR CONNECTIVITY CORRECTNESS
    fig3 = plt.figure(figsize=(5, 5))
    ax2d3 = fig3.add_subplot(111)

    s = np.sum(L)
    ax2d3.set_xlim(-s, s)
    ax2d3.set_ylim(-s, s)
    ax2d3.set_aspect('equal')
    ax2d3.axis('off')

    total_area = 0.0
    wedge_refs = []

    # ======================================================
    #                    MAIN LOOP
    # ======================================================
    for i in range(len(points)):
        x, y = points[i]
        beta_ranges, reliable_beta_ranges = compute_beta_range(x, y, L, CA)

        ranges_to_calculate_area = []

        for b_r_index in indices:
            b_r = reliable_beta_ranges[b_r_index]
            color = color_list[b_r_index]

            for beta_range in b_r:
                ranges_to_calculate_area.append(beta_range)

                theta1 = np.degrees(beta_range[0])
                theta2 = np.degrees(beta_range[1])
                outer_radius = x + ring_width / 2.0

                w = Wedge(
                    center=(0, 0),
                    r=outer_radius,
                    theta1=theta1,
                    theta2=theta2,
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                    alpha=1.0,
                )
                ax2d3.add_patch(w)
                wedge_refs.append(w)

        if ranges_to_calculate_area:
            merged_ranges = union_ranges(ranges_to_calculate_area)
            inner_r = x - ring_width / 2.0
            outer_r = x + ring_width / 2.0
            ring_area = np.pi * (outer_r**2 - inner_r**2)

            for b0, b1 in merged_ranges:
                total_area += ring_area * ((b1 - b0) / (2 * np.pi))

        del beta_ranges, reliable_beta_ranges, ranges_to_calculate_area

    # ======================================================
    #  Convert to grayscale image (this triggers rasterization)
    # ======================================================
    img_gray = figure_to_gray_image(fig3)

    # Compute connectivity
    connectivity_value = compute_connectivity(img_gray)

    # ======================================================
    #                ABSOLUTE CLEANUP (IMPORTANT)
    # ======================================================

    # Remove Wedges
    for w in wedge_refs:
        w.remove()
    wedge_refs.clear()

    # Clear axes artists
    ax2d3.cla()

    # Clear figure and renderer caches
    fig3.canvas.draw()          # ensures the renderer exists
    fig3.clf()                  # removes all artists
    fig3.canvas.flush_events()  # flush backend state

    # Close figure to release GUI/renderer resources
    plt.close(fig3)

    # Delete references to axes and fig
    del ax2d3
    del fig3

    # Force garbage collection AND free Python memory immediately
    gc.collect()
    gc.collect()

    return connectivity_value



#L =  [0.0001,1.49995,1.49995]
#CA = [(-2.8856310512880317, 2.8856310512880317), (-0.3672802511511213, 1.2181326405622919), (-3.0246087669394037, 1.878588173306368)]
#L =  [1.4644359570957466, 0.04698977532204848, 1.4885742675822051]
#CA = [(-1.4706632955724315, 1.4706632955724315), (-1.5768414877131869, 1.3524303899971144), (0.21929326925655435, 2.252802982198095)]
#L =  [1, 1, 1]
#CA = [(-18.2074 * np.pi / 180, 18.2074 * np.pi / 180), (-111.3415 * np.pi / 180, 111.3415 * np.pi / 180),(-111.3415 * np.pi / 180, 111.3415 * np.pi / 180)]
#L=[1.273685932707902, 0.47198624642931564, 1.2543278208627822]
#CA=[(-2.980810601260852, 2.980810601260852), (-2.2294043441860674, 2.9627856964286843), (1.6889140110258536, 1.9469783681522597)]

L=[0.5, 1.25, 1.25]
CA=[(-180*np.pi/180.0, 180*np.pi/180.0), (-53.1301*np.pi/180.0, 126.8698*np.pi/180.0), (106.2602*np.pi/180.0, 108.2602*np.pi/180.0)]
L2=[0.008849317757047144, 1.465181316077158, 1.525969366165795]
CA2=[(-3.104323811217068, 3.104323811217068), (-2.2135476717406846, -2.0770894991977737), (-3.0424740198157405, 0.028968517296451335)]
C_dot_A = CA.copy()
C_dot_A[0] = (-np.pi, np.pi)
connectivity_value=planar_3R_greyscale_connectivity_analysis(L,CA)
connectivity_value2=planar_3R_greyscale_connectivity_analysis(L2,CA2)
print(connectivity_value)
print(connectivity_value2)
print(connectivity_value/connectivity_value2)









