# Convert to NumPy array for easy manipulation
import ast
import json

import numpy as np
from matplotlib import pyplot as plt
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        list_of_lists = json.load(file)  # Load as list of lists
    return [np.array(lst, dtype=np.float64).reshape(3, 1) for lst in list_of_lists]

# File containing the data
filename = "plot_list.txt"
array_list = read_data_from_file(filename)
points = np.hstack(array_list)  # Shape will be (3, N)

# Extract theta values
theta1, theta2, theta3 = points


# Find min and max values for each theta
theta1_min, theta1_max = np.min(theta1), np.max(theta1)
theta2_min, theta2_max = np.min(theta2), np.max(theta2)
theta3_min, theta3_max = np.min(theta3), np.max(theta3)

extreme_points = []
for i in range(points.shape[1]):
    t1, t2, t3 = points[:, i]
    if t1 in [theta1_min, theta1_max] or t2 in [theta2_min, theta2_max] or t3 in [theta3_min, theta3_max]:
        extreme_points.append((t1, t2, t3))


# Assign colors for differentiation (Red, Green, Blue)
colors = ['r', 'g', 'b']
color_cycle = [colors[i % len(colors)] for i in range(len(extreme_points))]


# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(theta1, theta2, theta3, c='b', marker='o')

# Set labels
ax.set_xlabel("$\Theta_1$", fontsize=18)
ax.set_ylabel("$\Theta_2$", fontsize=18)
ax.set_zlabel("$\Theta_3$", fontsize=18)

# Set limits
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_zlim([-np.pi, np.pi])

# Default tick positions for Theta1, Theta2, and Theta3
default_ticks = [-np.pi, 0, np.pi]
default_labels = ["$-\pi$", "$0$", "$\pi$"]

# Function to add custom min/max ticks to an axis
def add_min_max_ticks(axis_ticks, axis_labels, min_val, max_val, min_label, max_label):
    if min_val not in axis_ticks:
        axis_ticks.append(min_val)
        axis_labels.append(min_label)
    if max_val not in axis_ticks:
        axis_ticks.append(max_val)
        axis_labels.append(max_label)
    # Sort the ticks in increasing order
    sorted_indices = np.argsort(axis_ticks)
    return np.array(axis_ticks)[sorted_indices], np.array(axis_labels)[sorted_indices]

# Modify Theta1 ticks
theta1_ticks, theta1_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta1_min, theta1_max, r"$\theta_1^{\text{min}}$", r"$\theta_1^{\text{max}}$"
)

# Modify Theta2 ticks
theta2_ticks, theta2_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta2_min, theta2_max, r"$\theta_2^{\text{min}}$", r"$\theta_2^{\text{max}}$"
)

# Modify Theta3 ticks
theta3_ticks, theta3_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta3_min, theta3_max, r"$\theta_3^{\text{min}}$", r"$\theta_3^{\text{max}}$"
)

# Apply tick settings
ax.set_xticks(theta1_ticks)
ax.set_xticklabels(theta1_labels, fontsize=12)

ax.set_yticks(theta2_ticks)
ax.set_yticklabels(theta2_labels, fontsize=12)

ax.set_zticks(theta3_ticks)
ax.set_zticklabels(theta3_labels, fontsize=12)

# Plot extreme points in different colors
for (t1, t2, t3), color in zip(extreme_points, color_cycle):
    ax.scatter(t1, t2, t3, color=color, s=100)  # Differentiate extreme points

# Show plot
plt.show()
