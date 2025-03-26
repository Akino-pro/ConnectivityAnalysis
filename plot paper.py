# Convert to NumPy array for easy manipulation
import ast
import json

import matplotlib
import numpy as np
import warnings
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
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
print(theta1_min, theta1_max ,theta2_min, theta2_max ,theta3_min, theta3_max)
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
ax.scatter(theta1, theta2, theta3, c='k', marker='o',s=10)

# Set labels
ax.set_xlabel(r"$\theta_1$", fontsize=18)
ax.set_ylabel(r"$\theta_2$", fontsize=18)
ax.set_zlabel(r"$\theta_3$", fontsize=18)

# Set limits
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_zlim([-np.pi, np.pi])

# Default tick positions for Theta1, Theta2, and Theta3
default_ticks = [-np.pi, 0, np.pi]
default_labels = ["$-\pi$", "$0$", "$\pi$"]

# Function to add custom min/max ticks to an axis
def add_min_max_ticks(axis_ticks, axis_labels, min_val, max_val, min_label, max_label, extra_ticks=[], extra_labels=[]):
    if min_val not in axis_ticks:
        axis_ticks.append(min_val)
        axis_labels.append(min_label)
    if max_val not in axis_ticks:
        axis_ticks.append(max_val)
        axis_labels.append(max_label)

    # Add extra ticks if they are not already included
    for tick, label in zip(extra_ticks, extra_labels):
        if tick not in axis_ticks:
            axis_ticks.append(tick)
            axis_labels.append(label)

    # Sort the ticks in increasing order
    sorted_indices = np.argsort(axis_ticks)
    return np.array(axis_ticks)[sorted_indices], np.array(axis_labels)[sorted_indices]


# Modify Theta1 ticks (Adding a_1^min, a_1^max)
theta1_ticks, theta1_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta1_min, theta1_max,
    r"$\theta̲_1$", r"$\overline{\theta}_1$",
    extra_ticks=[-0.5, 0.5],
    extra_labels=[r"$a̲_1$", r"$\overline{a}_1$"]
)

# Modify Theta2 ticks (Adding a^2_min, a^2_max)
theta2_ticks, theta2_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta2_min, theta2_max,
    r"$\theta̲_2$", r"$\overline{\theta}_2$",
    extra_ticks=[-2.61, -1],
    extra_labels=[r"$a̲_2$", r"$\overline{a}_2$"]
)

# Modify Theta3 ticks (Adding a_1^min, a_1^max for Theta3)
theta3_ticks, theta3_labels = add_min_max_ticks(
    list(default_ticks), list(default_labels),
    theta3_min, theta3_max,
    r"$\theta̲_3$", r"$\overline{\theta}_3$",
    extra_ticks=[-1, 1],
    extra_labels=[r"$a̲_3$", r"$\overline{a}_3$"]
)

# Apply tick settings
ax.set_xticks(theta1_ticks)
ax.set_xticklabels(theta1_labels, fontsize=12)

ax.set_yticks(theta2_ticks)
ax.set_yticklabels(theta2_labels, fontsize=12)

ax.set_zticks(theta3_ticks)
ax.set_zticklabels(theta3_labels, fontsize=12)

# Plot extreme points in different colors
#for (t1, t2, t3), color in zip(extreme_points, color_cycle):
#    ax.scatter(t1, t2, t3, color=color, s=100)  # Differentiate extreme points

# Apply colors to the extreme ticks
def color_ticks(axis, tick_values, min_val, max_val, min_color, max_color):
    tick_labels = axis.get_ticklabels()
    for label, tick in zip(tick_labels, tick_values):
        if tick == min_val:
            label.set_color(min_color)
        elif tick == max_val:
            label.set_color(max_color)
        else:
            label.set_color('black')  # Keep default ticks black

theta1_extreme_colors = {'min': 'g', 'max': 'g'}
theta2_extreme_colors = {'min': 'b', 'max': 'b'}
theta3_extreme_colors = {'min': 'r', 'max': 'r'}
# Color each extreme tick

color_ticks(ax.xaxis, theta1_ticks, theta1_min, theta1_max, theta1_extreme_colors['min'],theta1_extreme_colors['max'])
color_ticks(ax.yaxis, theta2_ticks, theta2_min, theta2_max, theta2_extreme_colors['min'],theta2_extreme_colors['max'])
color_ticks(ax.zaxis, theta3_ticks, theta3_min, theta3_max, theta3_extreme_colors['min'],theta3_extreme_colors['max'])

# Plot extreme points in different colors
for (t1, t2, t3), color in zip(extreme_points, color_cycle):
    ax.scatter(t1, t2, t3, color=color, s=100)  # Differentiate extreme points

# Add dashed blue lines from extreme values of theta2 to plane theta3=-π and π
for i, theta1_extreme in enumerate([theta1_min, theta1_max]):  # Loop over extreme theta2 values
    corresponding_theta2 = theta2[np.argwhere(theta1 == theta1_extreme).flatten()[0]]  # Get corresponding theta1
    corresponding_theta3 = theta3[np.argwhere(theta1 == theta1_extreme).flatten()[0]]
    ax.plot(
        [theta1_extreme, theta1_extreme],  # Use the theta1 value corresponding to the extreme theta2
        [corresponding_theta2, corresponding_theta2],  # Theta2 remains at extreme values
        [-np.pi, corresponding_theta3],  # Line segment from -π to π
        linestyle="dashed", color="green", linewidth=2
    )

    ax.plot(
        [theta1_extreme, theta1_extreme],  # Use the theta1 value corresponding to the extreme theta2
        [-np.pi,np.pi],  # Theta2 remains at extreme values
        [-np.pi, -np.pi],  # Line segment from -π to π
        linestyle="dashed", color="green", linewidth=2
    )


# Add dashed blue lines from extreme values of theta2 to plane theta3=-π and π
for i, theta2_extreme in enumerate([theta2_min, theta2_max]):  # Loop over extreme theta2 values
    corresponding_theta1 = theta1[np.argwhere(theta2 == theta2_extreme).flatten()[0]]  # Get corresponding theta1
    corresponding_theta3 = theta3[np.argwhere(theta2 == theta2_extreme).flatten()[0]]
    ax.plot(
        [corresponding_theta1, corresponding_theta1],  # Use the theta1 value corresponding to the extreme theta2
        [theta2_extreme, theta2_extreme],  # Theta2 remains at extreme values
        [-np.pi, corresponding_theta3],  # Line segment from -π to π
        linestyle="dashed", color="blue", linewidth=2
    )

    ax.plot(
        [-np.pi,np.pi],  # Use the theta1 value corresponding to the extreme theta2
        [theta2_extreme, theta2_extreme],  # Theta2 remains at extreme values
        [-np.pi, -np.pi],  # Line segment from -π to π
        linestyle="dashed", color="blue", linewidth=2
    )
# Add dashed blue lines from extreme values of theta2 to plane theta3=-π and π
for i, theta3_extreme in enumerate([theta3_min, theta3_max]):  # Loop over extreme theta2 values
    corresponding_theta1 = theta1[np.argwhere(theta3 == theta3_extreme).flatten()[0]]  # Get corresponding theta1
    corresponding_theta2 = theta2[np.argwhere(theta3 == theta3_extreme).flatten()[0]]

    ax.plot(
        [corresponding_theta1, corresponding_theta1],  # Use the theta1 value corresponding to the extreme theta2
        [corresponding_theta2, np.pi],  # Theta2 remains at extreme values
        [theta3_extreme, theta3_extreme],  # Line segment from -π to π
        linestyle="dashed", color="red", linewidth=2
    )


    ax.plot(
        [-np.pi, np.pi],  # Use the theta1 value corresponding to the extreme theta2
        [np.pi, np.pi],  # Theta2 remains at extreme values
        [theta3_extreme, theta3_extreme],  # Line segment from -π to π
        linestyle="dashed", color="red", linewidth=2
    )
# Show plot
plt.show()
