from tqdm import tqdm

from Planar_3r_FTW_morphological_estimation_discretization import planar_3R_connectivity_analysis
import numpy as np

from spatial_4r_FTW_morphological_estimation_ssm import ssm_estimation

sample_number = 128


def find_local_optimum(sample):
    step_size = 0.2
    searching_num = int(2 * np.pi / step_size)
    global_maximum = ssm_estimation(sample_number, sample['d'], sample['alpha'], sample['a'], sample['theta'])

    for i in range(4):
        local_winner = []
        local_maximum = global_maximum
        for j in tqdm(range(searching_num), desc="Processing Items"):
            if i == 0:
                new_try_lower = sample['theta'][i][0] + j * step_size
                if new_try_lower > np.pi:
                    new_try_lower -= 2 * np.pi
                elif new_try_lower < -np.pi:
                    new_try_lower += 2 * np.pi
                if new_try_lower <= 0:
                    new_jl = sample['theta'].copy()
                    new_jl[0] = (new_try_lower, -new_try_lower)
                    search_result = ssm_estimation(sample_number, sample['d'], sample['alpha'], sample['a'], new_jl)
                    if search_result >= local_maximum:
                        local_maximum = search_result
                        local_winner = [(new_try_lower, -new_try_lower)]
            else:
                new_try_lower = sample['theta'][i][0] + j * step_size
                if new_try_lower > np.pi:
                    new_try_lower -= 2 * np.pi
                elif new_try_lower < -np.pi:
                    new_try_lower += 2 * np.pi
                new_try_upper = sample['theta'][i][1]
                if new_try_lower <= new_try_upper:
                    new_jl = sample['theta'].copy()
                    new_jl[i] = (new_try_lower, new_try_upper)
                    search_result = ssm_estimation(sample_number, sample['d'], sample['alpha'], sample['a'], new_jl)
                    if search_result >= local_maximum:
                        local_maximum = search_result
                        local_winner.append((new_try_lower, new_try_upper))

                new_try_upper = sample['theta'][i][1] + j * step_size
                if new_try_upper > np.pi:
                    new_try_upper -= 2 * np.pi
                elif new_try_upper < -np.pi:
                    new_try_upper += 2 * np.pi
                new_try_lower = sample['theta'][i][0]
                if new_try_lower <= new_try_upper:
                    new_jl = sample['theta'].copy()
                    new_jl[i] = (new_try_lower, new_try_upper)
                    search_result = ssm_estimation(sample_number, sample['d'], sample['alpha'], sample['a'], new_jl)
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
        'd': [0, 1, -1, 0.5]
        ,
        'alpha': [1.5707963267948966, -1.5707963267948966, 1.5707963267948966, 0.0],
        'a': [1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 0.8660254037844386]
        ,
        'theta': [(-0.2411657969381082, 0.2411657969381082), (1.3847495872170423, 3.0344699871967173),
                  (-3.013074541299022, -0.20428201822201242), (-1.2707294073172148, 2.4655786159604465)]
        ,
    }
)
