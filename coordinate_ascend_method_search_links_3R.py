from tqdm import tqdm
import numpy as np

from planar_3R_greyscale_reliability_map import planar_3R_greyscale_connectivity_analysis


def generate_random_link_lengths(total_length=3.0):
    """
    Generate a random (L1, L2, L3) such that:
        L1 > 0, L2 > 0, L3 > 0, and L1 + L2 + L3 = total_length
    """
    cuts = np.sort(np.random.rand(2) * total_length)
    l1 = cuts[0]
    l2 = cuts[1] - cuts[0]
    l3 = total_length - cuts[1]
    return [l1, l2, l3]


def find_link_length_improvements(sample,
                                  num_trials=500,
                                  max_improvements=20,
                                  total_length=3.0):
    """
    Random search over link lengths with constraint L1 + L2 + L3 = total_length,
    keeping joint_limits fixed.

    Parameters
    ----------
    sample : dict
        {
            'link_lengths': [L1, L2, L3],
            'joint_limits': [(l1_min, l1_max),
                             (l2_min, l2_max),
                             (l3_min, l3_max)]
        }
    num_trials : int
        Maximum number of random candidates to evaluate.
    max_improvements : int
        Stop after finding this many improving link-length combinations.
    total_length : float
        Constraint L1 + L2 + L3 = total_length.

    Returns
    -------
    best_sample : dict
        Sample dict with the best link lengths found (and original joint limits).
    best_value : float
        Best connectivity value.
    improvements : list of dict
        List of all improving candidates:
        [
            {
                'link_lengths': [...],
                'value': connectivity_value
            },
            ...
        ]
    """

    # Fix joint limits; keep them exactly as in input
    fixed_joint_limits = [tuple(jl) for jl in sample['joint_limits']]

    # If the initial link lengths don't sum to total_length, we can normalize or just warn.
    init_L = np.array(sample['link_lengths'], dtype=float)
    if abs(np.sum(init_L) - total_length) > 1e-6:
        print(f"Warning: initial link lengths sum to {np.sum(init_L):.6f}, "
              f"not {total_length:.6f}. Using them anyway as the baseline.")

    # Evaluate starting point
    base_value = planar_3R_greyscale_connectivity_analysis(
        init_L,
        fixed_joint_limits
    )
    print(f"Initial link lengths: {init_L.tolist()}")
    print(f"Initial connectivity = {base_value}")

    best_value = base_value
    best_links = init_L.tolist()
    improvements = []

    desc = "Searching link length combinations"
    for _ in tqdm(range(num_trials), desc=desc):
        # Sample a new candidate (L1, L2, L3) with sum = total_length
        candidate_links = generate_random_link_lengths(total_length=total_length)

        val = planar_3R_greyscale_connectivity_analysis(
            candidate_links,
            fixed_joint_limits
        )

        # Check improvement (with small epsilon for float noise)
        if val > best_value + 1e-12:
            best_value = val
            best_links = candidate_links
            improvements.append({
                'link_lengths': candidate_links,
                'value': float(val),
            })

            print("\nFound improvement!")
            print("  New best link lengths:", candidate_links)
            print("  New best connectivity:", val)

            if len(improvements) >= max_improvements:
                print(f"\nReached max_improvements = {max_improvements}, stopping search.")
                break

    print("\nFinal best connectivity:", best_value)
    print("Final best link lengths:", best_links)
    print(f"Total improvements found: {len(improvements)}")

    best_sample = {
        'link_lengths': best_links,
        'joint_limits': fixed_joint_limits
    }

    return best_sample, best_value, improvements


# Example call
if __name__ == "__main__":
    start_sample = {
        'link_lengths': [0.008849317757047144, 1.465181316077158, 1.525969366165795],
        'joint_limits': [
            (-2.8856310512880317, 2.8856310512880317),
            (-0.3672802511511213, 1.2181326405622919),
            (-3.0246087669394037, 1.878588173306368),
        ],
    }

    best_sample, best_value, improvements = find_link_length_improvements(
        start_sample,
        num_trials=100,        # how many random candidates to try
        max_improvements=30,    # how many improved combinations to collect
        total_length=3.0        # L1 + L2 + L3 = 3
    )
