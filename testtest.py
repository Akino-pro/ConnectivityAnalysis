import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fibonacci_sphere_modified(n, r=1):
    """
    Generate vertices evenly distributed on the surface of a sphere using a
    modified Fibonacci Sphere Algorithm:

    - Uses the golden angle (2*pi / phi^2).
    - Includes an offset (0.5) in the index to avoid placing a point exactly at the pole.

    Args:
    - n (int): Number of points.
    - r (float): Radius of the sphere. Default is 1.

    Returns:
    - points (list of tuples): List of (x, y, z) coordinates on the sphere.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # The golden ratio, ~1.618
    # The "golden angle" often used in phyllotaxis:
    golden_angle = 2.0 * np.pi / (phi ** 2)  # ~2.399963

    points = []
    for i in range(n):
        # Offset from 0 to avoid the exact pole
        offset_i = i + 0.5
        # Map z from +1 to -1
        z = 1.0 - 2.0 * offset_i / n
        # Radius in x-y plane
        xy_radius = np.sqrt(1.0 - z * z)

        # Angle increments by the golden angle
        theta = golden_angle * offset_i

        x = xy_radius * np.cos(theta)
        y = xy_radius * np.sin(theta)

        points.append((r * x, r * y, r * z))

    return points


def plot_fibonacci_sphere_modified(n, r=1):
    """
    Plot the vertices generated using the modified Fibonacci Sphere.

    Args:
    - n (int): Number of points.
    - r (float): Radius of the sphere. Default is 1.
    """
    points = fibonacci_sphere_modified(n, r)
    x_vals, y_vals, z_vals = zip(*points)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_vals, y_vals, z_vals, color='b', s=10)

    ax.set_box_aspect((1, 1, 1))  # Ensures the sphere looks round
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Modified Fibonacci Sphere with {n} points, radius={r}")

    plt.show()


# Example usage
if __name__ == "__main__":
    plot_fibonacci_sphere_modified(100)
