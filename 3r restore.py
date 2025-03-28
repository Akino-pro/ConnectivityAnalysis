import numpy as np
import matplotlib

from Two_dimension_connectivity_measure import connectivity_analysis

matplotlib.use('Agg')  # Use Agg backend before importing pyplot
import matplotlib.pyplot as plt

from helper_functions import plot_workspace


def sample_points_circle(radius,num_samples):
    """
    Generates random points uniformly sampled inside a circle of given radius.

    Parameters:
        radius (float): Radius of the circle.
        num_samples (int): Number of points to sample.

    Returns:
        numpy.ndarray: Sampled (x, y) points of shape (num_samples, 2).
    """
    angles = np.random.uniform(0, 2 * np.pi, num_samples)  # Random angles
    radii = np.sqrt(np.random.uniform(0, 1, num_samples)) * radius  # Uniform sampling in a circle
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def sample_circle_and_plot(radius, num_samples):
    """
    Samples points inside a circle and returns the grayscale matrix of the plot.

    Parameters:
        radius (float): Radius of the circle.
        num_samples (int): Number of points to sample.
        data_point_size (int): Size of plotted points.

    Returns:
        numpy.ndarray: Greyscale image matrix from the plotted workspace.
    """
    workspace = sample_points_circle(radius, num_samples)
    grayscale_matrix = plot_workspace(workspace, color='k')
    return grayscale_matrix

# Example Usage
binary_image = sample_circle_and_plot(radius=3, num_samples=100000)

shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_image,
                                                                                     4, 0.5)
print(shape_area, connected_connectivity, general_connectivity)