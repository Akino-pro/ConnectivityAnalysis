import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Planar_3r_FTW_morphological_estimation_discretization import planar_3R_connectivity_analysis
from spatial_4r_FTW_morphological_estimation_ssm import ssm_estimation

pd.set_option('display.max_colwidth', None)  # No column width truncation
pd.set_option('display.width', 1000)  # Set display width to a large value

sample_number = 20
num_generations = 2
grid_sample_number = 72
alpha = 0.05


def normalize_angle(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def sample_and_normalize_range(start, end):
    start_norm = normalize_angle(start)
    end_norm = normalize_angle(end)

    if start_norm <= end_norm and (end - start <= 2 * np.pi):
        # The range fits within -pi to pi after normalization
        return start_norm, end_norm
    else:
        # Keep the range as-is since it doesn't map to -pi to pi uniquely
        return start, end


def generate_spatial_4r_parameters():
    # Sample each parameter based on the constraints provided
    params = {
        'alpha': [],
        'a': [],
        'd': [],
        'theta': []
    }

    for i in range(4):
        # Sample alpha_i within [-π/2, π/2]
        alpha_i = np.random.uniform(-np.pi / 2, np.pi / 2)
        params['alpha'].append(alpha_i)

        # Sample a_i within [0, 1]
        a_i = np.random.uniform(0, 1)
        params['a'].append(a_i)

        # Sample d_i within [-1, 1]
        d_i = np.random.uniform(-1, 1)
        params['d'].append(d_i)

        # Sample theta_i based on conditions
        if i == 0:
            # theta0 with symmetric limit
            upper_limit = np.random.uniform(0, np.pi)
            lower_limit = -upper_limit
            params['theta'].append((lower_limit, upper_limit))
        else:
            theta_a = np.random.uniform(-np.pi, np.pi)
            theta_b = np.random.uniform(-np.pi, np.pi)
            theta_li = min(theta_a, theta_b)
            theta_up = max(theta_a, theta_b)
            if theta_li > np.pi and theta_up > np.pi:
                theta_li -= 2 * np.pi
                theta_up -= 2 * np.pi
            if theta_li < -np.pi and theta_up < -np.pi:
                theta_li += 2 * np.pi
                theta_up += 2 * np.pi
            if theta_li <= np.pi <= theta_up:
                theta_a = np.random.uniform(-np.pi, np.pi)
                theta_b = np.random.uniform(-np.pi, np.pi)
                theta_li = min(theta_a, theta_b)
                theta_up = max(theta_a, theta_b)
            theta_tuple = (theta_li, theta_up)
            params['theta'].append(theta_tuple)

    # print(params)
    return params


def elitism(samples: pd.DataFrame, alpha=alpha):
    # Sort the dataframe by the last column (which we will ignore for the return)
    sorted_samples = samples.sort_values(by=samples.columns[-1], ascending=False)

    # Calculate the cutoff index for the top alpha percent
    cutoff = int(len(sorted_samples) * alpha)

    # Split into top alpha percent (to retain as is, excluding the last column)
    top_alpha = sorted_samples.head(cutoff).iloc[:, :-1]

    new_children = []
    # Generate new population for the bottom 95% using stochastic selection and two-point crossover
    remaining_population_size = len(sorted_samples) - cutoff
    top_alpha_values = top_alpha.values.tolist()

    while len(new_children) < remaining_population_size:
        # Select two parents from the entire sample pool (not just top 5%)
        parent1, parent2 = stochastic_selection(samples.iloc[:, :-1])  # Pass only the 9 variables for crossover

        # Generate two children using two-point crossover
        child1, child2 = uniform_crossover(parent1, parent2)
        child1_list = [child1['alpha'], child1['a'], child1['d'], child1['theta']]
        child2_list = [child2['alpha'], child2['a'], child2['d'], child2['theta']]
        if child1 not in new_children and child1_list not in top_alpha_values:
            new_children.append(child1)
        if child2 not in new_children and child2_list not in top_alpha_values and len(
                new_children) < remaining_population_size:
            new_children.append(child2)

    # Convert the list of children into a DataFrame and concatenate with top_alpha
    new_children_population = pd.DataFrame(new_children, columns=samples.columns[:-1])

    return top_alpha, new_children_population


def stochastic_selection(samples):
    """Select two parents randomly from the entire sample set, which contains two lists (link_lengths and joint_limits)."""
    # Select two parents as rows, each containing 'link_lengths' and 'joint_limits'
    parent1 = samples.sample().iloc[0].to_dict()  # Convert the row to a dictionary
    parent2 = samples.sample().iloc[0].to_dict()  # Convert the row to a dictionary
    return parent1, parent2


def uniform_crossover(parent1, parent2, seed=None):
    """
    Apply uniform crossover to two parents by randomly choosing the entire list
    of each parameter from one of the two parents to create two offspring.

    Parameters:
    - parent1 (dict): Dictionary with lists as values for the first parent.
    - parent2 (dict): Dictionary with lists as values for the second parent.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - child1 (dict): First offspring with parameters chosen as entire lists from parent1 or parent2.
    - child2 (dict): Second offspring with parameters chosen as entire lists from parent1 or parent2.
    """
    if seed is not None:
        np.random.seed(seed)

    child1, child2 = {}, {}

    # For each parameter, randomly select the entire list from either parent1 or parent2
    for param in parent1.keys():
        if np.random.rand() > 0.5:
            child1[param] = parent1[param]
            child2[param] = parent2[param]
        else:
            child1[param] = parent2[param]
            child2[param] = parent1[param]

    return child1, child2


def mutate_child(child):
    """Apply mutation to a child by randomly reinitializing genes with 0.01 probability."""
    # Mutate the entire child if mutation occurs
    # if True:

    # Generate new random parameters similar to creating a new sample
    new_params = generate_spatial_4r_parameters()
    new_child = child.copy()

    # Update the child's values with the new parameters
    new_child[0] = new_params['alpha']
    new_child[1] = new_params['a']
    new_child[2] = new_params['d']
    new_child[3] = new_params['theta']

    return new_child


def apply_mutation(population: pd.DataFrame, top_5_percent, mutation_rate: float = 0.15):
    """Apply mutation to 15% of the children."""
    # Select 15% of the children for mutation
    num_to_mutate = int(len(population) * mutation_rate)
    children_to_mutate = population.sample(num_to_mutate)
    top_alpha_values = top_5_percent.values.tolist()

    # Mutate each selected child
    mutated_population = population.copy()
    population_list = mutated_population.values.tolist()
    # print(population_list)
    # unique_population_set = set(
    # tuple(tuple(row) if isinstance(row, list) else row for row in r) for r in population.values.tolist())
    for idx in children_to_mutate.index:
        child = mutated_population.loc[idx].values
        mutated_child = mutate_child(child)
        mutated_child_list = [
            mutated_child[0],  # The first element is already a list
            mutated_child[1],
            mutated_child[2],
            [list(item) for item in mutated_child[3]]  # Convert each tuple to a list in the second element
        ]
        while mutated_child_list in population_list or mutated_child_list in top_alpha_values:
            mutated_child = mutate_child(mutated_child)
        mutated_population.loc[idx] = mutated_child
        population_list = mutated_population.values.tolist()
        # mutated_tuple = tuple(tuple(mutated_child[0]) + tuple(tuple(limit) for limit in mutated_child[1]))
        # if mutated_tuple not in unique_population_set:
        # unique_population_set.add(mutated_tuple)
        # mutated_population.loc[idx] = mutated_child

    return mutated_population


def get_champion(samples: pd.DataFrame, global_elite_limit):
    """
    Returns the first row (champion) from a dataframe of samples sorted by the last column.
    """
    # Sort the dataframe by the last column in descending order
    # samples=connectivity_analysis(samples)
    sorted_samples = samples.sort_values(by=samples.columns[-1], ascending=False)
    new_global_elite_limit = global_elite_limit
    champion_limit = int(len(sorted_samples) * alpha)
    # Return the first row, which is the champion
    champion = sorted_samples.head(champion_limit)

    print(champion)
    if champion.iloc[-1]['connectivity'] > new_global_elite_limit:
        new_global_elite_limit = champion.iloc[-1]['connectivity']
    # champion_history.append(champion.iloc[0]['connectivity'])
    # champion_history.append(champion)

    return champion, new_global_elite_limit


def connectivity_analysis(samples, global_elite_limit):
    # Initialize an empty list to store the sum for each row

    connectivity_values = []
    elites = [global_elite_limit]
    limit_size = int(sample_number * 0.05)

    # Iterate over each row in the dataframe
    for index, row in samples.iterrows():
        print(row)
        connectivity_value = ssm_estimation(grid_sample_number, row['d'], row['alpha'], row['a'], row['theta'])
        if connectivity_value > elites[len(elites) - 1]:
            if len(elites) < limit_size:
                elites.append(connectivity_value)
            else:
                elites[len(elites) - 1] = connectivity_value
            elites = sorted(elites, reverse=True)
        connectivity_values.append(connectivity_value)

    # Add the computed sums as a new column named 'connectivity'
    samples['connectivity'] = connectivity_values

    champion, new_global_elite_limit = get_champion(samples, global_elite_limit)
    print(champion['alpha'])
    print(champion['a'])
    print(champion['d'])
    print(champion['theta'])
    print(champion['connectivity'])

    return samples, new_global_elite_limit


# Function to generate next generation
def next_generation(samples, global_elite_limit, alpha=0.05):
    samples, new_global_elite_limit = connectivity_analysis(samples, global_elite_limit)

    # Apply elitism and generate new children population (excluding the last column)
    top_5_percent, new_children_population = elitism(samples, alpha=alpha)

    # Apply mutation to the new children population (95% of the original size)
    mutated_population = apply_mutation(new_children_population, top_5_percent, mutation_rate=0.15)

    # Combine the top 5% with the mutated population to form the next generation
    next_gen = pd.concat([top_5_percent, mutated_population], ignore_index=True)

    return next_gen, new_global_elite_limit


prior_knowledge = [
    {
        'alpha': [85 * np.pi / 180, -53 * np.pi / 180, -89 * np.pi / 180, 68 * np.pi / 180],
        'a': [0.5, 0.48, 0.76, 0.95],
        'd': [-0.29, 0, 0.05, 1],
        'theta': [(-146 * np.pi / 180, 146 * np.pi / 180), (-234 * np.pi / 180, 10 * np.pi / 180),
                  (-115 * np.pi / 180, 132 * np.pi / 180), (-101 * np.pi / 180, 118 * np.pi / 180)]
    },
]
# Generate given number of samples
samples = prior_knowledge[:]
additional_samples_needed = sample_number - len(prior_knowledge)
if additional_samples_needed > 0:
    samples += [generate_spatial_4r_parameters() for _ in range(additional_samples_needed)]

# Display the samples in a dataframe
df = pd.DataFrame(samples)
current_generation = df

# print(current_generation)
global_elite_limit = 0
for i in range(num_generations):
    # Perform a single step of the genetic algorithm
    current_generation, new_global_elite_limit = next_generation(current_generation, global_elite_limit, alpha=0.05)
    global_elite_limit = new_global_elite_limit
    print(f"Generation {i} completed,{i + 1}generated.")
current_generation, new_global_elite_limit = connectivity_analysis(current_generation, global_elite_limit)
