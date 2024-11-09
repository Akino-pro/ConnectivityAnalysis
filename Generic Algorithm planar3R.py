import math
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Planar_3r_FTW_morphological_estimation_discretization import planar_3R_connectivity_analysis

pd.set_option('display.max_colwidth', None)  # No column width truncation
pd.set_option('display.width', 1000)  # Set display width to a large value

# num_threads=4
sample_number = 140
num_generations = 20
pfs = 20
alpha = 0.05


# champion_history = []


def generate_planar_3r_params():
    # Generate random link lengths such that l1 + l2 + l3 = 3 and l1, l2, l3 > 0
    lengths = np.sort(np.random.rand(2) * 3)  # Scale by 3
    l1 = lengths[0]
    l2 = lengths[1] - lengths[0]
    l3 = 3 - lengths[1]

    link_lengths = [l1, l2, l3]
    joint_limits = []
    for i in range(3):
        if i == 0:
            upper_limit = np.random.uniform(0, np.pi)
            lower_limit = -upper_limit
        else:
            lower_limit = np.random.uniform(-np.pi, np.pi)
            upper_limit = np.random.uniform(lower_limit, np.pi)  # Ensure lower <= upper
        joint_limits.append((lower_limit, upper_limit))

    return {
        'link_lengths': link_lengths,
        'joint_limits': joint_limits
    }


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
        child1, child2 = two_point_crossover(parent1, parent2)
        child1_list = [child1['link_lengths'], child1['joint_limits']]
        child2_list = [child2['link_lengths'], child2['joint_limits']]
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


def two_point_crossover(parent1, parent2):
    """Apply crossover by swapping link lengths and joint limits between two parents."""
    # Create two children by swapping the link_lengths and joint_limits between the parents
    child1 = {
        'link_lengths': parent1['link_lengths'],  # Link lengths from parent1
        'joint_limits': parent2['joint_limits']  # Joint limits from parent2
    }

    child2 = {
        'link_lengths': parent2['link_lengths'],  # Link lengths from parent2
        'joint_limits': parent1['joint_limits']  # Joint limits from parent1
    }

    return child1, child2


def mutate_child(child):
    """Apply mutation to a child by randomly reinitializing genes with 0.01 probability."""
    # Mutate the entire child if mutation occurs
    # if True:

    # Generate new random parameters similar to creating a new sample
    new_params = generate_planar_3r_params()
    new_child = child.copy()

    # Update the child's values with the new parameters
    new_child[0] = new_params['link_lengths']  # Update link_lengths (first entry)
    new_child[1] = new_params['joint_limits']  # Update joint_limits (second entry)

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
            [list(item) for item in mutated_child[1]]  # Convert each tuple to a list in the second element
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
        connectivity_value = planar_3R_connectivity_analysis(row['link_lengths'], row['joint_limits'], pfs,
                                                             elites[len(elites) - 1])
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
    print(champion['link_lengths'])
    print(champion['joint_limits'])
    print(champion['connectivity'])

    return samples, new_global_elite_limit


"""

def process_batch(batch, elites, pfs):
    connectivity_values = []
    for index, row in batch.iterrows():
        connectivity_value = planar_3R_connectivity_analysis(row['link_lengths'], row['joint_limits'], pfs, elites[-1])
        connectivity_values.append(connectivity_value)
    return connectivity_values

def update_elites(elites, connectivity_values, limit_size):
    for connectivity_value in connectivity_values:
        if connectivity_value > elites[-1]:
            if len(elites) < limit_size:
                elites.append(connectivity_value)
            else:
                elites[-1] = connectivity_value
            elites.sort(reverse=True)
    return elites

def connectivity_analysis(samples, num_threads=num_threads, batch_size=num_threads):
    # Initialize variables
    connectivity_values = []
    elites = [0]
    limit_size = int(len(samples) * 0.05)  # 5% of total samples

    # Create ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Process the samples in batches
        future_to_batch = {}
        for i in range(0, len(samples), batch_size):
            batch = samples.iloc[i:i + batch_size]
            future = executor.submit(process_batch, batch, elites, pfs)
            future_to_batch[future] = batch

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_connectivity_values = future.result()
            connectivity_values.extend(batch_connectivity_values)
            elites = update_elites(elites, batch_connectivity_values, limit_size)

    # Add the computed connectivity values to the samples DataFrame
    samples['connectivity'] = connectivity_values

    # Identify the champion
    champion = get_champion(samples)
    print(champion['link_lengths'])
    print(champion['joint_limits'])
    print(champion['connectivity'])

    return samples
"""


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


"""
prior_knowledge=[{
        'link_lengths':  [0.4888656043245976, 1.3992499610293656, 1.1118844346460368]
        ,
        'joint_limits':   [(-3.141592653589793, 3.141592653589793), (-0.9272951769138392, 2.2142957313467018), (1.8545903538276785, 1.8545903538276785)]
    }]
"""
prior_knowledge = [

    {
        'link_lengths':   [0.4888656043245976, 1.3992499610293656, 1.1118844346460368]
        ,
        'joint_limits': [(-3.141592653589793, 3.141592653589793), (-0.9272951769138392, 2.2142957313467018), (1.8545903538276785, 1.8545903538276785)]
    },
    {
        'link_lengths': [0.5,1.25,1.25]
        ,
        'joint_limits': [(-3.141592653589793, 3.141592653589793), (-0.9272951769138392, 2.2142957313467018), (1.8545903538276785, 1.8545903538276785)]

    },
    {
        'link_lengths':   [0.0409381072168612, 1.60650255325608, 1.3525593395270588]
        ,
        'joint_limits':   [(-2.12720203873519, 2.12720203873519), (-2.949995660344648, 1.169447348312247), (-2.935115465988839, 0.4973433364265576)]
    },
    {
        'link_lengths':    [0.008519589482239076, 1.87637511701138, 1.1151052935063808]
        ,
        'joint_limits':   [(-1.6341880196716376, 1.6341880196716376), (-2.994127677428315, -0.3414440400066612), (-1.3320745205876303, 2.9100694769133835)]
    }
]

#prior_knowledge = []
# Generate given number of samples
samples = prior_knowledge[:]
additional_samples_needed = sample_number - len(prior_knowledge)
if additional_samples_needed > 0:
    samples += [generate_planar_3r_params() for _ in range(additional_samples_needed)]
# samples = [generate_planar_3r_params() for _ in range(sample_number)]

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

# print(champion_history)

# generations = list(range(1, len(champion_history) + 1))
# plt.plot(generations, champion_history, marker='o', linestyle='-', color='b')

# Adding labels and title
# plt.xlabel('Generation Number')
# plt.ylabel('Connectivity')
# plt.title('Connectivity vs. Generation')

# Saving the plot to a file
# plt.grid(True)
# plt.savefig('connectivity_plot.png')
