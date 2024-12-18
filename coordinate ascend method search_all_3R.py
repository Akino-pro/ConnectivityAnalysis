from tqdm import tqdm

from Planar_3r_FTW_morphological_estimation_discretization import planar_3R_connectivity_analysis
import numpy as np


def find_local_optimum(sample):
    step_size = 0.1
    searching_num = int(2 * np.pi / step_size)
    global_maximum = planar_3R_connectivity_analysis(sample['link_lengths'], sample['joint_limits'], 20, 0)

    for i in range(3):
        local_winner = []
        local_maximum = global_maximum
        for j in tqdm(range(searching_num), desc="Processing Items"):
            if i == 0:
                new_try_lower = sample['joint_limits'][i][0] + j * step_size
                if new_try_lower > np.pi:
                    new_try_lower -= 2 * np.pi
                elif new_try_lower < -np.pi:
                    new_try_lower += 2 * np.pi
                if new_try_lower <= 0:
                    new_jl = sample['joint_limits'].copy()
                    new_jl[0] = (new_try_lower, -new_try_lower)
                    search_result = planar_3R_connectivity_analysis(sample['link_lengths'], new_jl, 20, 0)
                    if search_result >= local_maximum:
                        local_maximum = search_result
                        local_winner = [(new_try_lower, -new_try_lower)]
            else:
                new_try_lower = sample['joint_limits'][i][0] + j * step_size
                if new_try_lower > np.pi:
                    new_try_lower -= 2 * np.pi
                elif new_try_lower < -np.pi:
                    new_try_lower += 2 * np.pi
                new_try_upper = sample['joint_limits'][i][1]
                if new_try_lower <= new_try_upper:
                    new_jl = sample['joint_limits'].copy()
                    new_jl[i] = (new_try_lower, new_try_upper)
                    search_result = planar_3R_connectivity_analysis(sample['link_lengths'], new_jl, 20, 0)
                    if search_result >= local_maximum:
                        local_maximum = search_result
                        local_winner.append((new_try_lower, new_try_upper))

                new_try_upper = sample['joint_limits'][i][1] + j * step_size
                if new_try_upper > np.pi:
                    new_try_upper -= 2 * np.pi
                elif new_try_upper < -np.pi:
                    new_try_upper += 2 * np.pi
                new_try_lower = sample['joint_limits'][i][0]
                if new_try_lower <= new_try_upper:
                    new_jl = sample['joint_limits'].copy()
                    new_jl[i] = (new_try_lower, new_try_upper)
                    search_result = planar_3R_connectivity_analysis(sample['link_lengths'], new_jl, 20, 0)
                    if search_result >= local_maximum:
                        local_maximum = search_result
                        local_winner.append((new_try_lower, new_try_upper))
        print(f'search along joint {i} completed!')
        if local_maximum != global_maximum:
            print(f'found a better result!')
            print(local_maximum)
            print(local_winner)


find_local_optimum(
    {
        'link_lengths': [1,1,1]
        ,
        'joint_limits': [(-3.031883452592004, 3.031883452592004), (-1.619994146091692, -0.8276157453255935), (-1.6977602095460234, -0.7265946655975718)]
    }
)
