import threading
import time  # Import time module for time measurement
import matplotlib

from helper_functions import plot_workspace

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Two_dimension_connectivity_measure import connectivity_analysis

# import cupy as cp

# print(cv2.getBuildInformation())
# if not cv2.cuda.getCudaEnabledDeviceCount():
# print("not supported for GPU")
# parameters for FTW estimation
sample_density = 150  # correlated to accuracy of estimation but increase complexity
data_point_size = 15  # use to fill out the blank space between data points.
# parameters for connectivity estimation
kernel_size = 4
Lambda = 0.5
r1 = 0.5  # prob of joint 1 having issue.
r2 = 0.6
r3 = 0.7
all_workspace = []


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


def reliability_computation(r1, r2, r3):
    #reliability_list = [r2 * r3, r1 * r3, r1 * r2, r2 * r3 + r1 * r3 - r1 * r2 * r3,
    #                    r1 * r2 + r2 * r3 - r1 * r2 * r3, r1 * r2 + r1 * r3 - r1 * r2 * r3,
    #                    r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3]
    reliability_list = [r2 * r3 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r2 * r3 - r1 * r2 * r3, r1 * r2 + r1 * r3 - r1 * r2 * r3,
                        r1 * r2 + r1 * r3 + r2 * r3 - 2 * r1 * r2 * r3]
    conditional_reliability_list = []
    for p in reliability_list:
        conditional_reliability_list.append((p - r1 * r2 * r3) / (r1 * r2 + r1 * r3 + r2 * r3 - 3 * r1 * r2 * r3))
    return conditional_reliability_list


# now only consider 7 different region with different probability of confidence
cr_list = reliability_computation(r1, r2, r3)
#print(cr_list )

# print(cr_list  )


# Function to compute intersection of binary images
# @measure_time
def intersect_binary_images(binary_images):
    """
    Perform pixel-wise intersection (AND operation) on a list of binary images.

    Parameters:
    binary_images : List of binary images (numpy arrays).

    Returns:
    intersection_image : The resulting binary image after performing intersection.
    """
    if len(binary_images) == 0:
        image_size = (800, 800)
        return np.zeros(image_size, dtype=np.uint8)
    if len(binary_images) == 1:
        return binary_images[0]
    intersection_image = binary_images[0]
    for img in binary_images[1:]:
        intersection_image = cv2.bitwise_and(intersection_image, img)
    return intersection_image


def forward_kinematics_3R(theta1, theta2, theta3, L):
    """
    Computes the forward kinematics for a 3R planar robot arm.

    Parameters:
    theta1, theta2, theta3 : Joint angles in radians
    L : List containing the lengths of the three links

    Returns:
    (x, y) : Tuple containing the end-effector position (x, y)
    """
    x1 = L[0] * np.cos(theta1)
    y1 = L[0] * np.sin(theta1)

    x2 = x1 + L[1] * np.cos(theta1 + theta2)
    y2 = y1 + L[1] * np.sin(theta1 + theta2)

    x3 = x2 + L[2] * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + L[2] * np.sin(theta1 + theta2 + theta3)

    return (x3, y3)


# Configuration space sampling within artificial joint limits
# @measure_time
def compute_workspace(L, joint_limits, resolution=sample_density):
    theta_ranges = []
    for i in range(3):
        if joint_limits[i][0] == joint_limits[i][1]:
            theta_ranges.append([joint_limits[i][0]])  # Fixed joint
        else:
            theta_ranges.append(np.linspace(joint_limits[i][0], joint_limits[i][1], resolution))  # Variable joint

    theta1_range, theta2_range, theta3_range = theta_ranges

    workspace = []

    for theta1 in theta1_range:
        for theta2 in theta2_range:
            for theta3 in theta3_range:
                x, y = forward_kinematics_3R(theta1, theta2, theta3, L)
                workspace.append([x, y])

    return np.array(workspace)




# Compute the pre-failure workspace with 1.5x resolution

# Compute post-failure workspaces by locking each joint
# Compute post-failure workspaces by locking each joint
# @measure_time
def compute_post_failure_workspace(L, joint_limits, locked_joint_idx, pfs, championship, resolution=sample_density):
    """
    Compute the workspace after locking one joint.

    Parameters:
    L : List containing the lengths of the three links
    joint_limits : List of tuples containing joint limits in radians
    locked_joint_idx : Index of the joint to be locked (0, 1, or 2)
    resolution : Number of samples for the remaining joints

    Returns:
    workspace : List of (x, y) positions representing the post-failure workspace
    """
    post_failure_workspaces = []

    for locked_angle in np.linspace(joint_limits[locked_joint_idx][0], joint_limits[locked_joint_idx][1], pfs):
        # Allow remaining joints to move between full physical limits [-pi, pi]
        remaining_joint_limits = [(-np.pi, np.pi) if i != locked_joint_idx else (locked_angle, locked_angle) for i in
                                  range(len(joint_limits))]
        # print(remaining_joint_limits)
        workspace_for_theta = compute_workspace(L, remaining_joint_limits, int(resolution))
        grayscale_matrix = plot_workspace(workspace_for_theta)
        # Convert the plot to grayscale matrix and process it
        # grayscale_matrix = convert_plot_to_grayscale_matrix()
        # plt.close()
        processed_img = cv2.bitwise_not(grayscale_matrix.astype(np.uint8))
        # shape_area = np.sum(processed_img == 255)
        # if shape_area == 0 or shape_area < championship:
        #    return intersect_binary_images([])
        # cv2.imshow(f"post_failure_workspace fixed joint {locked_joint_idx + 1}", processed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        post_failure_workspaces.append(processed_img)
    post_failure_workspace = intersect_binary_images(post_failure_workspaces)
    # cv2.imshow(f"post_failure_workspace fixed joint {locked_joint_idx + 1}", post_failure_workspace)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return post_failure_workspace


start_time = time.time()

# @measure_time
def planar_3R_connectivity_analysis(L, joint_limits, pfs, championship):
    # for jm in joint_limits:
    #    print(jm[0]*180/np.pi)
    #    print(jm[1] * 180 / np.pi)
    pre_failure_workspace = compute_workspace(L, joint_limits, resolution=int(sample_density * 0.5))
    all_workspace.append(pre_failure_workspace)
    workspaces = []
    # Plot the pre-failure workspace using solid points
    grayscale_matrix = plot_workspace(pre_failure_workspace)
    # grayscale_matrix = convert_plot_to_grayscale_matrix()
    # plt.close()
    # grayscale_matrix = workspace_to_binary_matrix_with_size(pre_failure_workspace, data_point_size)
    processed_img = cv2.bitwise_not(grayscale_matrix.astype(np.uint8))
    shape_area = np.sum(processed_img == 255)
    # print(championship)
    if shape_area == 0 or shape_area < championship:
        print("too small")
        # return shape_area
        return 0
    # else:
    # cv2.imshow(f"post_failure_workspace fixed joint ",  processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    workspaces.append(processed_img)

    for i in range(3):
        post_failure_workspace = compute_post_failure_workspace(L, joint_limits, i, pfs, championship)
        shape_area = np.sum(post_failure_workspace == 255)
        if shape_area == 0 or shape_area < championship:
            print('too small')
            # return shape_area
            return 0
        workspaces.append(post_failure_workspace)
    fault_tolerant_workspace = intersect_binary_images(workspaces)

    # End total time measurement
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total execution time: {total_time:.4f} seconds")

    # Display the final intersected binary image using OpenCV
    cv2.imshow("Fault-Tolerant Workspace", fault_tolerant_workspace)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #shape_area = np.sum(fault_tolerant_workspace == 255)
    shape_area, connected_connectivity, general_connectivity = connectivity_analysis(fault_tolerant_workspace,
                                                                                     kernel_size, Lambda)
    print(shape_area)
    print(connected_connectivity)
    print(general_connectivity)
    # return connected_connectivity
    #return shape_area
    return general_connectivity




"""
# @measure_time
def planar_3R_connectivity_analysis(L, joint_limits, pfs, championship):
    pre_failure_workspace = compute_workspace(L, joint_limits, resolution=int(sample_density * 0.5))
    all_workspace.append(pre_failure_workspace)
    workspaces = []
    # Plot the pre-failure workspace using solid points
    grayscale_matrix = plot_workspace(pre_failure_workspace)
    # grayscale_matrix = convert_plot_to_grayscale_matrix()
    # plt.close()
    # grayscale_matrix = workspace_to_binary_matrix_with_size(pre_failure_workspace, data_point_size)
    processed_img = cv2.bitwise_not(grayscale_matrix.astype(np.uint8))
    shape_area = np.sum(processed_img == 255)
    # print(championship)
    # if shape_area == 0 or shape_area < championship:
    #    print("too small")
    #    return shape_area
    # else:
    # cv2.imshow(f"post_failure_workspace fixed joint ",  processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    workspaces.append(processed_img)

    for i in range(3):
        post_failure_workspace = compute_post_failure_workspace(L, joint_limits, i, pfs, championship)
        # shape_area = np.sum(post_failure_workspace == 255)
        # if shape_area == 0 or shape_area < championship:
        #    print('too small')
        #   return shape_area
        workspaces.append(post_failure_workspace)
    workspaces_cases = []
    #workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[1]]))
    #workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[2]]))
    #workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[3]]))
    workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[1], workspaces[2]]))
    workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[1], workspaces[3]]))
    workspaces_cases.append(intersect_binary_images([workspaces[0], workspaces[2], workspaces[3]]))
    workspaces_cases.append(intersect_binary_images(workspaces))
    reliable_connectivity = 0
    for index, wp_combination in enumerate(workspaces_cases):
        # if index == 6:
        #cv2.imshow("Fault-Tolerant Workspace", wp_combination)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        shape_area, connected_connectivity, general_connectivity = connectivity_analysis(wp_combination,
                                                                                         kernel_size, Lambda)
        #print(general_connectivity)
        reliable_connectivity += general_connectivity * cr_list[index]
    print(f'under probability {r1, r2, r3}the reliable connectivity of given configuration is {reliable_connectivity}!')

    return reliable_connectivity

"""

#planar_3R_connectivity_analysis([1.4142135623730951, 1.4142135623730951, 0.816496580927726],
#                               [(-3.031883452592004, 3.031883452592004), (-1.619994146091692, -0.8276157453255935), (-1.6977602095460234, -0.7265946655975718)], pfs=20, championship=0)
#planar_3R_connectivity_analysis([1.0, 1.0, 1.0],
#                                [(-3.031883452592004, 3.031883452592004), (-1.619994146091692, -0.8276157453255935), (-1.6977602095460234, -0.7265946655975718)], pfs=60,
 #                               championship=0)
planar_3R_connectivity_analysis([1,0, 1.0, 1.0],
                                [(-18.2074 * np.pi / 180, 18.2074 * np.pi / 180),
                                 (-111.3415 * np.pi / 180, 111.3415 * np.pi / 180),
                                (-111.3415 * np.pi / 180, 111.3415 * np.pi / 180)], pfs=60,
                                championship=0)


