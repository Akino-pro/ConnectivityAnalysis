import json
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from file
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        list_of_lists = json.load(file)  # Load as list of lists
    return [np.array(lst, dtype=np.float64).reshape(3, 1) for lst in list_of_lists]

# File containing the data
filename = "plot_list.txt"
array_list = read_data_from_file(filename)
points = np.hstack(array_list)  # Shape will be (3, N)

# Extract theta values
_, theta2, theta3 = points  # Ignore theta3

# Find min and max values for theta1 and theta2
theta2_min, theta2_max = np.min(theta2), np.max(theta2)
theta3_min, theta3_max = np.min(theta3), np.max(theta3)

# Create 2D plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(theta2, theta3, c='k', marker='o', s=10)  # Plot all points in black

# Identify extreme points
extreme_points = []
for i in range(points.shape[1]):
    t2, t3 = theta2[i], theta3[i]
    if t2 in [theta2_min, theta2_max] or t3 in [theta3_min, theta3_max]:
        extreme_points.append((t2, t3))

# Separate extreme points into \(\theta_1\) and \(\theta_2\) categories
extreme_theta2 = [(t2, t3) for t2, t3 in extreme_points if t2 in [theta2_min, theta2_max]]
extreme_theta3 = [(t2, t3) for t2, t3 in extreme_points if t3 in [theta3_min, theta3_max]]



if extreme_theta2:
    t2_vals, t3_vals = zip(*extreme_theta2)
    ax.scatter(t2_vals, t3_vals, c='green', marker='o', s=100, label=r"$\theta_2^{\text{extreme}}$")
    # Add dashed lines to the x-axis
    for t2, t3 in extreme_theta2:
        ax.plot([t2, t2], [t3, -np.pi], linestyle="dashed", color="green", linewidth=1.5, alpha=0.7)

if extreme_theta3:
    t2_vals, t3_vals = zip(*extreme_theta3)
    ax.scatter(t2_vals, t3_vals, c='red', marker='o', s=100, label=r"$\theta_3^{\text{extreme}}$")
    # Add dashed lines to the y-axis
    for t2, t3 in extreme_theta3:
        ax.plot([t2, -np.pi], [t3, t3], linestyle="dashed", color="red", linewidth=1.5, alpha=0.7)

# Set labels
ax.set_xlabel(r"$\theta_2$", fontsize=30)
ax.set_ylabel(r"$\theta_3$", fontsize=30)

# Set limits
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])

# Default tick positions and labels
default_ticks = [-np.pi, 0, np.pi]
default_labels = [r"$-\pi$", r"$0$", r"$\pi$"]

# Extra ticks and labels for theta1
extra_ticks_theta3 = [-1, 1]
extra_labels_theta3 = [r"$\text{𝒂̲}_3$", r"$\text{𝒂̅}_3$"]

# Extra ticks and labels for theta2
extra_ticks_theta2 = [-2.61, -1]
extra_labels_theta2 = [r"$\text{𝒂̲}_2$", r"$\text{𝒂̅}_2$"]

# Add extreme values and extra ticks for theta1
theta2_ticks = default_ticks + [theta2_min, theta2_max] + extra_ticks_theta2
theta2_labels = default_labels + [r"$\theta̲_{2_\text{min}}$", r"$\overline{\theta}_{2_\text{max}}$"] + extra_labels_theta2

# Add extreme values and extra ticks for theta2
theta3_ticks = default_ticks + [theta3_min, theta3_max] + extra_ticks_theta3
theta3_labels = default_labels + [r"$\theta̲_{3_\text{min}}$", r"$\overline{\theta}_{3_\text{max}}$"] + extra_labels_theta3
ax.set_aspect('equal')
# Apply tick settings
ax.set_xticks(theta2_ticks)
ax.set_xticklabels(theta2_labels, fontsize=25)

ax.set_yticks(theta3_ticks)
ax.set_yticklabels(theta3_labels, fontsize=25)

# Function to color specific ticks
def color_ticks(axis, tick_values, min_val, max_val, extra_ticks, min_color, max_color, extra_color):
    for label, tick in zip(axis.get_ticklabels(), tick_values):
        if tick == min_val or tick == max_val:
            label.set_color(min_color)
        elif tick in extra_ticks:
            label.set_color(extra_color)
        else:
            label.set_color('black')  # Default color for regular ticks

# Color extreme and extra ticks
color_ticks(ax.xaxis, theta2_ticks, theta2_min, theta2_max, extra_ticks_theta2, 'green', 'green', 'k')
color_ticks(ax.yaxis, theta3_ticks, theta3_min, theta3_max, extra_ticks_theta3, 'red', 'red', 'k')

ax.fill_betweenx(
    y=np.linspace(extra_ticks_theta3[0], extra_ticks_theta3[1], 100),  # Range in theta3
    x1=extra_ticks_theta2[0],
    x2=extra_ticks_theta2[1],
    color='blue',
    alpha=0.1
)


# Show plot
ax.grid(True, linestyle="--", alpha=0.6)
plt.show()
