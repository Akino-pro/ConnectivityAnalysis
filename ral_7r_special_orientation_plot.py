import ast

from helper_functions import fibonacci_sphere_angles, normalize_and_map_colors, plot_voronoi_regions_on_sphere

max_length = 4.461932111892875
with open("my_7r_list.txt", "r") as file:
    content = file.read()
    all_data = ast.literal_eval(content)
orientation_samples=len(all_data[0][2])
theta_phi_list = fibonacci_sphere_angles(orientation_samples)



color_list_ori, sm_ori = normalize_and_map_colors(all_data[0][5])
plot_voronoi_regions_on_sphere(theta_phi_list,
                                   color_list_ori,
                                   sm_ori
                                   )