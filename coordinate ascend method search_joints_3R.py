from tqdm import tqdm
import numpy as np

from planar3R_reliable_connectivity_analysis import planar_3r_reliable_connectivity_analysis
from planar_3R_greyscale_reliability_map import planar_3R_greyscale_connectivity_analysis


def wrap_to_pi(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find_local_optimum(sample):
    step_size = 0.1
    searching_num = int(2 * np.pi / step_size)

    # Make a working copy so we can mutate joint limits
    best_sample = {
        'link_lengths': list(sample['link_lengths']),
        'joint_limits': [tuple(jl) for jl in sample['joint_limits']]
    }

    # Evaluate starting point
    global_maximum = planar_3r_reliable_connectivity_analysis(
        best_sample['link_lengths'],
        best_sample['joint_limits']
    )

    print(f"Initial connectivity = {global_maximum}")

    # Coordinate search over each joint
    for i in range(3):
        local_maximum = global_maximum
        local_winners = []

        desc = f"Searching joint {i}"
        for j in tqdm(range(searching_num), desc=desc):
            # Base joint limits for this trial
            base_limits = [tuple(jl) for jl in best_sample['joint_limits']]

            if i == 0:
                # Joint 1: symmetric limits (-a, a)
                new_lower = base_limits[0][0] + j * step_size
                new_lower = wrap_to_pi(new_lower)

                # enforce symmetry and lower <= 0
                if new_lower <= 0:
                    new_upper = -new_lower
                    trial_limits = base_limits.copy()
                    trial_limits[0] = (new_lower, new_upper)

                    val = planar_3r_reliable_connectivity_analysis(
                        best_sample['link_lengths'],
                        trial_limits
                    )

                    if val > local_maximum + 1e-12:  # small epsilon to avoid float noise
                        local_maximum = val
                        local_winners = [(new_lower, new_upper)]
                    elif abs(val - local_maximum) <= 1e-12:
                        # equally good solution; keep as additional candidate
                        local_winners.append((new_lower, new_upper))

            else:
                # Joint 2 or 3: move lower and upper independently
                lower0, upper0 = base_limits[i]

                # ---- move lower bound ----
                new_lower = wrap_to_pi(lower0 + j * step_size)
                new_upper = upper0

                if new_lower <= new_upper:
                    trial_limits = base_limits.copy()
                    trial_limits[i] = (new_lower, new_upper)

                    val = planar_3r_reliable_connectivity_analysis(
                        best_sample['link_lengths'],
                        trial_limits
                    )

                    if val > local_maximum + 1e-12:
                        local_maximum = val
                        local_winners = [(new_lower, new_upper)]
                    elif abs(val - local_maximum) <= 1e-12:
                        local_winners.append((new_lower, new_upper))

                # ---- move upper bound ----
                new_upper = wrap_to_pi(upper0 + j * step_size)
                new_lower = lower0

                if new_lower <= new_upper:
                    trial_limits = base_limits.copy()
                    trial_limits[i] = (new_lower, new_upper)

                    val = planar_3r_reliable_connectivity_analysis(
                        best_sample['link_lengths'],
                        trial_limits
                    )

                    if val > local_maximum + 1e-12:
                        local_maximum = val
                        local_winners = [(new_lower, new_upper)]
                    elif abs(val - local_maximum) <= 1e-12:
                        local_winners.append((new_lower, new_upper))

        print(f"Search along joint {i} completed!")

        # If we improved, update the best_sample and global_maximum
        if local_maximum > global_maximum + 1e-12 and local_winners:
            print("Found a better result on joint", i)
            print("New best connectivity:", local_maximum)
            print("Best joint limits candidates:", local_winners)

            # For now, just pick the first winner
            chosen = local_winners[0]
            best_sample['joint_limits'][i] = chosen
            global_maximum = local_maximum

    print("Final best connectivity:", global_maximum)
    print("Final joint limits:", best_sample['joint_limits'])

    return best_sample, global_maximum


# Example call
best_sample, best_value = find_local_optimum(
    {
        'link_lengths': [1, 1, 1],
        'joint_limits':[(-3.141592653589793, 3.141592653589793), (0.640270911503988, 1.57313244849884), (0.4332034212492015, 1.6973158592745994)],
    }
)

