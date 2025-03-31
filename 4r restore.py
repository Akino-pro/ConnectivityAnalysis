import numpy as np

from Three_dimension_connectivity_measure import connectivity_analysis
from spatial3R_ftw_draw import generate_binary_matrix

N = 5000
nz = int(np.sqrt(2 * N))
nx = int(nz / 2)
grid_size = (64, 64, 64)
max_length=3.1989682240512938
x_range = (0, max_length)  # Range for x-axis
z_range = (-max_length, max_length)
# Create a list of [-pi, pi] repeated nx * nz times
pi_list = [[(-np.pi, np.pi)] for _ in range(nx * nz)]

binary_matrix, x_edges, y_edges, z_edges = generate_binary_matrix(
    nx , nz , x_range, z_range, grid_size, pi_list
)
shape_area, connected_connectivity, general_connectivity = connectivity_analysis(binary_matrix,
                                                                                         1, 0.5)
print(shape_area, connected_connectivity, general_connectivity)
v=np.pi*max_length*max_length*max_length*2
print(v)