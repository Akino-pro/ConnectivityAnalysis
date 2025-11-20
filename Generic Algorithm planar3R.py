import numpy as np
import pandas as pd
from planar_3R_greyscale_reliability_map import planar_3R_greyscale_connectivity_analysis

pd.set_option('display.max_colwidth', None)  # No column width truncation
pd.set_option('display.width', 1000)        # Set display width to a large value

# ============================================
# GA Hyperparameters
# ============================================
sample_number = 100
num_generations = 1
alpha = 0.05            # Top α fraction kept by elitism
mutation_rate = 0.15    # Fraction of children to mutate

# champion_history = []  # If you want to track best fitness per generation


# ============================================
# Parameter generation for planar 3R
# ============================================
def generate_planar_3r_params():
    """
    Generate random link lengths l1, l2, l3 such that:
        l1 > 0, l2 > 0, l3 > 0, and l1 + l2 + l3 = 3
    by sampling uniformly from a 2-simplex (Dirichlet distribution).

    Joint limits are generated as before.
    """

    # ---- Random link lengths satisfying l1 + l2 + l3 = 3 ----
    # Sample from Dirichlet(1,1,1) = uniform over simplex
    raw_lengths = np.random.dirichlet([1, 1, 1])
    link_lengths = (3 * raw_lengths).tolist()   # scale so sum = 3

    # ---- Joint limits ----
    joint_limits = []
    for i in range(3):
        if i == 0:
            # Joint 1 symmetric around zero
            upper_limit = np.random.uniform(0, np.pi)
            lower_limit = -upper_limit
        else:
            lower_limit = np.random.uniform(-np.pi, np.pi)
            upper_limit = np.random.uniform(lower_limit, np.pi)
        joint_limits.append((lower_limit, upper_limit))

    return {
        'link_lengths': link_lengths,
        'joint_limits': joint_limits
    }


# ============================================
# GA Operators
# ============================================
def elitism(samples: pd.DataFrame, alpha: float = alpha):
    """
    Select the top α fraction as elites and generate a new child population
    by stochastic selection + two-point crossover over the whole population.
    """
    # Sort by connectivity (last column)
    sorted_samples = samples.sort_values(by=samples.columns[-1], ascending=False)

    # Ensure at least one elite
    cutoff = max(1, int(len(sorted_samples) * alpha))

    # Top α as elites (drop connectivity column)
    top_alpha = sorted_samples.head(cutoff).iloc[:, :-1]

    # Children count = population size - elites
    remaining_population_size = len(sorted_samples) - cutoff

    new_children = []
    top_alpha_values = top_alpha.values.tolist()

    while len(new_children) < remaining_population_size:
        parent1, parent2 = stochastic_selection(samples.iloc[:, :-1])  # use only parameters

        child1, child2 = two_point_crossover(parent1, parent2)

        child1_list = [child1['link_lengths'], child1['joint_limits']]
        child2_list = [child2['link_lengths'], child2['joint_limits']]

        if child1_list not in top_alpha_values and child1_list not in new_children:
            new_children.append(child1_list)
        if len(new_children) < remaining_population_size:
            if child2_list not in top_alpha_values and child2_list not in new_children:
                new_children.append(child2_list)

    # Convert list-of-lists into DataFrame
    new_children_population = pd.DataFrame(new_children, columns=samples.columns[:-1])

    return top_alpha, new_children_population


def stochastic_selection(samples: pd.DataFrame):
    """
    Randomly select two parents from the sample DataFrame that has columns:
    'link_lengths', 'joint_limits' (no connectivity column).
    """
    parent1 = samples.sample().iloc[0].to_dict()
    parent2 = samples.sample().iloc[0].to_dict()
    return parent1, parent2


def two_point_crossover(parent1, parent2):
    """
    For this problem, crossover is implemented as swapping link_lengths
    and joint_limits between parents.
    """
    child1 = {
        'link_lengths': parent1['link_lengths'],
        'joint_limits': parent2['joint_limits']
    }
    child2 = {
        'link_lengths': parent2['link_lengths'],
        'joint_limits': parent1['joint_limits']
    }
    return child1, child2


def mutate_child():
    """
    Create a completely new child with random parameters.
    Returns [link_lengths, joint_limits] to match DataFrame row structure.
    """
    new_params = generate_planar_3r_params()
    return [new_params['link_lengths'], new_params['joint_limits']]


def apply_mutation(population: pd.DataFrame,
                   top_5_percent: pd.DataFrame,
                   mutation_rate: float = mutation_rate):
    """
    Apply mutation to a fraction of the children (population, which excludes elites).
    We ensure that mutated children are not duplicates of existing population
    or elites by comparing the list representation.
    """
    num_to_mutate = int(len(population) * mutation_rate)
    if num_to_mutate == 0:
        return population

    mutated_population = population.copy()

    # Represent rows as lists for membership checks
    population_list = mutated_population.values.tolist()
    top_alpha_values = top_5_percent.values.tolist()

    # Randomly choose indices to mutate
    children_to_mutate = mutated_population.sample(num_to_mutate).index

    for idx in children_to_mutate:
        candidate = mutate_child()

        # Avoid duplicates
        while candidate in population_list or candidate in top_alpha_values:
            candidate = mutate_child()

        mutated_population.at[idx, 'link_lengths'] = candidate[0]
        mutated_population.at[idx, 'joint_limits'] = candidate[1]

        # Update population_list to reflect this change
        population_list = mutated_population.values.tolist()

    return mutated_population


def get_champion(samples: pd.DataFrame, global_elite_limit):
    """
    Select the elite subset (top α fraction) and update global_elite_limit.
    global_elite_limit here is interpreted as the minimum connectivity of the
    best elite band seen so far (i.e., worst of the best).
    """
    sorted_samples = samples.sort_values(by=samples.columns[-1], ascending=False)

    champion_limit = max(1, int(len(sorted_samples) * alpha))
    champion = sorted_samples.head(champion_limit)

    new_global_elite_limit = global_elite_limit
    # Use the last (worst) of the elites to update the band threshold
    if champion.iloc[-1]['connectivity'] > new_global_elite_limit:
        new_global_elite_limit = champion.iloc[-1]['connectivity']

    # champion_history.append(champion.iloc[0]['connectivity'])

    print("Current elite band (top {}%):".format(int(alpha * 100)))
    print(champion)

    return champion, new_global_elite_limit


# ============================================
# Connectivity evaluation
# ============================================
def connectivity_analysis(samples: pd.DataFrame, global_elite_limit):
    """
    Evaluate connectivity for each sample using planar_3R_connectivity_analysis.
    Maintain an elite band (list of top connectivity values) and return the
    updated DataFrame and new global_elite_limit.
    """
    connectivity_values = []
    elites = [global_elite_limit]
    limit_size = int(sample_number * alpha) if int(sample_number * alpha) > 0 else 1

    for index, row in samples.iterrows():
        """
        connectivity_value = planar_3R_connectivity_analysis(
            row['link_lengths'],
            row['joint_limits'],
            pfs,
            elites[-1]
        )
        """
        connectivity_value = planar_3R_greyscale_connectivity_analysis(row['link_lengths'], row['joint_limits'])

        if connectivity_value > elites[-1]:
            if len(elites) < limit_size:
                elites.append(connectivity_value)
            else:
                elites[-1] = connectivity_value
            elites = sorted(elites, reverse=True)

        connectivity_values.append(connectivity_value)

    samples['connectivity'] = connectivity_values

    champion, new_global_elite_limit = get_champion(samples, global_elite_limit)
    print("Champion link lengths:\n", champion['link_lengths'])
    print("Champion joint limits:\n", champion['joint_limits'])
    print("Champion connectivity:\n", champion['connectivity'])

    return samples, new_global_elite_limit


def next_generation(samples: pd.DataFrame, global_elite_limit, alpha: float = alpha):
    """
    Single GA step:
      1. Evaluate connectivity
      2. Apply elitism (keep top α)
      3. Generate children via crossover
      4. Mutate some children
      5. Combine elites + mutated children
    """
    samples, new_global_elite_limit = connectivity_analysis(samples, global_elite_limit)

    # Elitism: top α kept as-is, rest replaced by offspring
    top_5_percent, new_children_population = elitism(samples, alpha=alpha)

    # Mutate a fraction of the children
    mutated_population = apply_mutation(new_children_population, top_5_percent, mutation_rate=mutation_rate)

    # New generation = elites + children
    next_gen = pd.concat([top_5_percent, mutated_population], ignore_index=True)

    return next_gen, new_global_elite_limit


# ============================================
# Prior knowledge (optional seeds)
# ============================================
"""
prior_knowledge = [{
    'link_lengths': [1, 1, 1],
    'joint_limits': [(-0.7491244590957556, 0.7491244590957556),
                     (-0.2590887855330002, 1.5556037937159994),
                     (-2.11211741668124, 2.1202051003030054)]
},
 ...
]
"""

prior_knowledge = []


# ============================================
# Main GA loop
# ============================================
if __name__ == "__main__":
    # Initialize population with optional prior knowledge
    samples = prior_knowledge[:]
    additional_samples_needed = sample_number - len(prior_knowledge)
    if additional_samples_needed > 0:
        samples += [generate_planar_3r_params() for _ in range(additional_samples_needed)]

    df = pd.DataFrame(samples)   # columns: link_lengths, joint_limits
    current_generation = df

    global_elite_limit = 0.0

    for i in range(num_generations):
        print(f"=== Generation {i} ===")
        current_generation, global_elite_limit = next_generation(
            current_generation,
            global_elite_limit,
            alpha=alpha
        )
        print(f"Generation {i} completed, {i + 1} generated.\n")

    # Final evaluation on last generation
    current_generation, global_elite_limit = connectivity_analysis(
        current_generation,
        global_elite_limit
    )

    # If you re-enable champion_history, you can plot:
    # generations = list(range(1, len(champion_history) + 1))
    # plt.plot(generations, champion_history, marker='o', linestyle='-')
    # plt.xlabel('Generation Number')
    # plt.ylabel('Connectivity')
    # plt.title('Connectivity vs. Generation')
    # plt.grid(True)
    # plt.savefig('connectivity_plot.png')
