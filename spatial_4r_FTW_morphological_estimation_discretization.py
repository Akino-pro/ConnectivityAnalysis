import random
import time  # Import time module for time measurement
import matplotlib

matplotlib.use('TkAgg')
# matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Three_dimension_connectivity_measure import connectivity_analysis
from skimage.measure import label

# import cupy as cp

# print(cv2.getBuildInformation())
# if not cv2.cuda.getCudaEnabledDeviceCount():
# print("not supported for GPU")

# parameters for FTW estimation
sample_density = 100  # correlated to accuracy of estimation but increase complexity
data_point_size = 15  # use to fill out the blank space between data points.
# parameters for connectivity estimation
kernel_size = 1
Lambda = 0.5


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


def on_key(event):
    if event.key == 'q':  # Press 'q' to close the figure (like waitKey behavior)
        plt.close(event.canvas.figure)


def random_color():
    return [random.random() for _ in range(3)]


@measure_time
def plot_3d_object(labeled_object, num_components, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for component in range(1, num_components + 1):
        # Get the coordinates of the current component
        x, y, z = np.nonzero(labeled_object == component)

        # Plot the current component with a random color
        ax.scatter(x, y, z, color=random_color(), s=data_point_size, marker='o', label=f'Component {component}')
    ax.set_xlim([0, labeled_object.shape[0]])
    ax.set_ylim([0, labeled_object.shape[1]])
    ax.set_zlim([0, labeled_object.shape[2]])
    ax.set_title(title)
    ax.set_axis_off()
    fig.canvas.mpl_connect('key_press_event', on_key)
    # plt.show()


def dh_transformation_matrix(theta, d, a, alpha):
    """
    Computes the individual transformation matrix using DH parameters.

    Parameters:
    theta : Joint angle in radians
    d : Offset along the previous z-axis to the common normal
    a : Link length (distance along the common normal to the previous z-axis)
    alpha : Twist angle (angle between z-axes)

    Returns:
    T : 4x4 Transformation matrix
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T


def forward_kinematics_3R_DH(dh_params):
    """
    Computes the forward kinematics for a 4R robot arm using DH parameters.

    Parameters:
    dh_params : List of DH parameters for the four joints.
                Each element in the list is a tuple (theta, d, a, alpha)

    Returns:
    end_effector_position : The (x, y, z) position of the end-effector.
    """

    # Start with the identity matrix
    T_total = np.eye(4)

    # Multiply transformation matrices for each joint
    for params in dh_params:
        T_joint = dh_transformation_matrix(*params)
        T_total = np.dot(T_total, T_joint)  # Matrix multiplication

    # Extract the end-effector position (x, y, z) from the final transformation matrix
    x = T_total[0, 3]
    y = T_total[1, 3]
    z = T_total[2, 3]
    return (x, y, z)


@measure_time
def convert_3d_plot_to_binary_voxel_matrix(ax, grid_size=100, plot_range=(-4, 4)):
    """
    Convert a 3D plot to a binary 3D matrix (voxel grid), considering a plot range of [-4, 4].

    Parameters:
    ax : Matplotlib 3D axis object that contains the 3D plot data.
    grid_size : The size of the 3D grid (number of voxels along each axis), default is 100x100x100.
    plot_range : Tuple representing the range of the plot along each axis (default is (-4, 4)).

    Returns:
    voxel_matrix : A binary 3D matrix (voxel grid) representing the 3D points from the plot.
    """
    # Get the data points from the 3D plot (scatter)
    points = ax.collections[0]._offsets3d  # Extract the 3D scatter plot points

    # Convert the scatter plot data (x, y, z) into arrays
    x = np.array(points[0])
    y = np.array(points[1])
    z = np.array(points[2])

    # Create an empty 3D matrix with the given size (initialized to 0)
    voxel_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Get the plot range min and max values for normalization
    plot_min, plot_max = plot_range

    # Normalize x, y, and z values to fit within the voxel grid dimensions
    if np.max(x) != np.min(x):
        x_norm = np.clip(((x - plot_min) / (plot_max - plot_min) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    else:
        x_norm = np.zeros_like(x, dtype=int)

    if np.max(y) != np.min(y):
        y_norm = np.clip(((y - plot_min) / (plot_max - plot_min) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    else:
        y_norm = np.zeros_like(y, dtype=int)

    if np.max(z) != np.min(z):
        z_norm = np.clip(((z - plot_min) / (plot_max - plot_min) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    else:
        z_norm = np.zeros_like(z, dtype=int)

    # Mark the corresponding (x, y, z) positions in the 3D matrix as "on" (1) for data points
    for i in range(len(x_norm)):
        voxel_matrix[x_norm[i], y_norm[i], z_norm[i]] = 1  # "On" voxel for data points

    return voxel_matrix


# Function to compute intersection of binary images
@measure_time
def intersect_binary_objects(binary_objects):
    """
    Perform pixel-wise intersection (AND operation) on a list of binary images.

    Parameters:
    binary_images : List of binary images (numpy arrays).

    Returns:
    intersection_image : The resulting binary image after performing intersection.
    """
    if len(binary_objects) == 0:
        object_size = (800, 800, 800)
        return np.zeros(object_size, dtype=np.uint8)
    if len(binary_objects) == 1:
        return binary_objects[0]
    intersection_object = binary_objects[0]
    for object in binary_objects[1:]:
        intersection_object = np.logical_and(intersection_object, object)
    return intersection_object


# Configuration space sampling within artificial joint limits
@measure_time
def compute_workspace_3R(a, joint_limits, d, alpha, resolution=sample_density):
    """
    Compute the reachable workspace by sampling the configuration space within joint limits.

    Parameters:
    L : List containing the lengths of the four links
    joint_limits : List of tuples containing joint limits in radians
    resolution : Number of samples for each joint angle

    Returns:
    workspace : List of (x, y, z) positions representing the workspace
    """

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
                dh_params = [
                    (theta1, d[0], a[0], alpha[0]),  # Joint 1 parameters
                    (theta2, d[1], a[1], alpha[1]),  # Joint 2 parameters
                    (theta3, d[2], a[2], alpha[2]),  # Joint 3 parameters
                ]
                x, y, z = forward_kinematics_3R_DH(dh_params)
                workspace.append([x, y, z])

    return np.array(workspace)


@measure_time
# Function to plot workspace using solid points
def plot_workspace_3D(workspace, color='k', p_mode=False):
    """
    Plot the workspace in 3D using solid points.

    Parameters:
    workspace : List or array of (x, y, z) positions representing the workspace
    color : Color of the workspace points
    data_point_size : Size of the plotted data points
    """
    # Create a 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the (x, y, z) points
    ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2],
               color=color, s=data_point_size)

    # Labels for the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set the range of the axes explicitly to [-4, 4]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # Add grid and disable coordinate frame axes
    plt.grid(True)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    if p_mode:
        plt.show()
    plt.close(fig)
    return ax


@measure_time
def compute_post_failure_workspace_3R(a, joint_limits, d, alpha, locked_joint_idx, pfs, championship,
                                      resolution=sample_density):
    """
    Compute the workspace after locking one joint for the 4R robot.

    Parameters:
    L : List containing the lengths of the four links
    joint_limits : List of tuples containing joint limits in radians
    locked_joint_idx : Index of the joint to be locked (0, 1, 2, or 3)
    resolution : Number of samples for the remaining joints

    Returns:
    workspace : List of (x, y) positions representing the post-failure workspace
    """
    post_failure_workspaces = []

    for locked_angle in np.linspace(joint_limits[locked_joint_idx][0], joint_limits[locked_joint_idx][1], pfs):
        remaining_joint_limits = [(-np.pi, np.pi) if i != locked_joint_idx else (locked_angle, locked_angle) for i in
                                  range(3)]
        workspace_for_theta = compute_workspace_3R(a, remaining_joint_limits, d, alpha, int(resolution * 0.6))
        ax = plot_workspace_3D(workspace_for_theta)
        grayscale_matrix = convert_3d_plot_to_binary_voxel_matrix(ax)
        labeled_object, num_components = label(grayscale_matrix, connectivity=1, return_num=True)
        # plot_3d_object(labeled_object, num_components, 'Fault-Tolerant Workspace')
        shape_area = np.sum(grayscale_matrix.astype(np.uint8))

        if shape_area == 0 or shape_area < championship:
            return intersect_binary_objects([])

        post_failure_workspaces.append(grayscale_matrix)

    post_failure_workspace = intersect_binary_objects(post_failure_workspaces)
    return post_failure_workspace


start_time = time.time()


def three_dimensional_3R_connectivity_analysis(a, joint_limits, d, alpha, pfs, championship):
    """
    Perform connectivity analysis for a 4R robot's workspace.

    Parameters:
    L : Link lengths
    joint_limits : Joint limits
    d : Offsets for DH parameters
    a : Link lengths for DH parameters
    pfs : Post-failure sampling density
    championship : Minimum workspace area threshold

    Returns:
    general_connectivity : The general connectivity of the fault-tolerant workspace
    """
    pre_failure_workspace = compute_workspace_3R(a, joint_limits, d, alpha, resolution=int(sample_density * 0.5))
    ax = plot_workspace_3D(pre_failure_workspace)
    grayscale_matrix = convert_3d_plot_to_binary_voxel_matrix(ax)
    # labeled_object, num_components = label(grayscale_matrix, connectivity=1, return_num=True)
    # plot_3d_object(labeled_object, num_components, 'Fault-Tolerant Workspace')
    plot_workspace_3D(grayscale_matrix, p_mode=True)
    shape_area = np.sum(grayscale_matrix.astype(np.uint8))

    if shape_area == 0 or shape_area < championship:
        print("Too small workspace")
        return shape_area

    workspaces = [grayscale_matrix]

    for i in range(3):
        post_failure_workspace = compute_post_failure_workspace_3R(a, joint_limits, d, alpha, i, pfs, championship)
        shape_area = np.sum(post_failure_workspace)

        if shape_area == 0 or shape_area < championship:
            print('Too small workspace')
            return shape_area

        workspaces.append(post_failure_workspace)

    fault_tolerant_workspace = intersect_binary_objects(workspaces)
    # plot_workspace_3D(fault_tolerant_workspace, p_mode=True)
    # labeled_object, num_components = label(fault_tolerant_workspace, connectivity=1, return_num=True)
    # plot_3d_object(labeled_object, num_components, 'Fault-Tolerant Workspace')

    shape_area, connected_connectivity, general_connectivity = connectivity_analysis(fault_tolerant_workspace,
                                                                                     kernel_size, Lambda)
    return general_connectivity


"""
dh_params = [
    (np.pi / 4, 0.1, 1, 0),   # Joint 1 parameters (theta1, d1, a1, alpha1)
    (np.pi / 6, 0.2, 1, 0),   # Joint 2 parameters (theta2, d2, a2, alpha2)
    (np.pi / 3, 0.3, 1, 0),   # Joint 3 parameters (theta3, d3, a3, alpha3)
]
"""

#three_dimensional_3R_connectivity_analysis([1.32, 1.27, 1]
#                                           , [(-146 * np.pi / 180, 146 * np.pi / 180), (-160 * np.pi / 180, 10 * np.pi / 180),
 #     (-115 * np.pi / 180, 132 * np.pi / 180), (-101 * np.pi / 180, 118 * np.pi / 180)]
 #                                          , [2.42, -0.38, 0.95]
#                                           , [-92* np.pi / 180, -93 * np.pi / 180, 0 * np.pi / 180]
#                                           , pfs=20, championship=0)
