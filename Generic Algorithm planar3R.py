import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from planar3R_reliable_connectivity_analysis import planar_3r_reliable_connectivity_analysis

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# ============================================
# GA Hyperparameters
# ============================================
sample_number = 128
num_generations = 1
alpha = 0.05
mutation_rate = 0.15            # adaptive later
immigrant_fraction = 0.05       # mild random restart
POPULATION_FILE = "population_saved.txt"


# ============================================
# Parameter generation for planar 3R
# ============================================
def generate_planar_3r_params():
    raw_lengths = np.random.dirichlet([1, 1, 1])
    link_lengths = (3 * raw_lengths).tolist()

    joint_limits = []
    for i in range(3):
        if i == 0:
            upper = np.random.uniform(0, np.pi)
            joint_limits.append((-upper, upper))
        else:
            lo = np.random.uniform(-np.pi, np.pi)
            hi = np.random.uniform(lo, np.pi)
            joint_limits.append((lo, hi))

    return {
        'link_lengths': link_lengths,
        'joint_limits': joint_limits
    }


# ============================================
# Chromosome extraction / enforcing constraints
# ============================================
def extract_chromosome(parent):
    l1, l2, _ = parent['link_lengths']

    _, jl1_up = parent['joint_limits'][0]
    u1 = jl1_up

    jl2_low, jl2_up = parent['joint_limits'][1]
    jl3_low, jl3_up = parent['joint_limits'][2]

    return np.array([l1, l2, u1,
                     jl2_low, jl2_up,
                     jl3_low, jl3_up])


def enforce_constraints(chrom):
    l1 = max(chrom[0], 1e-6)
    l2 = max(chrom[1], 1e-6)
    l3 = 3 - (l1 + l2)

    if l3 <= 0:
        total = l1 + l2
        l1 = 3 * (l1 / total)
        l2 = 3 * (l2 / total)
        l3 = 3 - (l1 + l2)

    # Joint 1
    u1 = np.clip(chrom[2], 0.01, np.pi)
    jl1_low = -u1
    jl1_up = u1

    # Joint 2
    jl2_low = np.clip(chrom[3], -np.pi, np.pi)
    jl2_up = np.clip(chrom[4], -np.pi, np.pi)
    if jl2_low > jl2_up:
        jl2_low, jl2_up = jl2_up, jl2_low

    # Joint 3
    jl3_low = np.clip(chrom[5], -np.pi, np.pi)
    jl3_up = np.clip(chrom[6], -np.pi, np.pi)
    if jl3_low > jl3_up:
        jl3_low, jl3_up = jl3_up, jl3_low

    return {
        'link_lengths': [l1, l2, l3],
        'joint_limits': [
            (jl1_low, jl1_up),
            (jl2_low, jl2_up),
            (jl3_low, jl3_up)
        ]
    }


# ============================================
# BLX-α crossover
# ============================================
def blx_alpha_crossover(parent1, parent2, alpha=0.3):
    p1 = extract_chromosome(parent1)
    p2 = extract_chromosome(parent2)

    child1 = np.zeros_like(p1)
    child2 = np.zeros_like(p1)

    for i in range(len(p1)):
        mn = min(p1[i], p2[i])
        mx = max(p1[i], p2[i])
        diff = mx - mn

        low = mn - alpha * diff
        high = mx + alpha * diff

        child1[i] = np.random.uniform(low, high)
        child2[i] = np.random.uniform(low, high)

    return enforce_constraints(child1), enforce_constraints(child2)


# ============================================
# Parent selection (stochastic)
# ============================================
def stochastic_selection(df):
    p1 = df.sample().iloc[0].to_dict()
    p2 = df.sample().iloc[0].to_dict()
    return p1, p2


# ============================================
# Mutate by replacing individual with random one
# ============================================
def mutate_child():
    new_params = generate_planar_3r_params()
    return [new_params['link_lengths'], new_params['joint_limits']]


def apply_mutation(children_df, elites_df, mutation_rate):
    num_mutate = int(len(children_df) * mutation_rate)
    if num_mutate == 0:
        return children_df

    mutated = children_df.copy()
    elite_list = elites_df.values.tolist()
    pop_list = mutated.values.tolist()

    indices = mutated.sample(num_mutate).index

    for idx in indices:
        c = mutate_child()
        while c in pop_list or c in elite_list:
            c = mutate_child()
        mutated.at[idx, 'link_lengths'] = c[0]
        mutated.at[idx, 'joint_limits'] = c[1]
        pop_list = mutated.values.tolist()

    return mutated


# ============================================
# Random immigrants (5% of population)
# ============================================
def inject_immigrants(children_df):
    k = max(1, int(len(children_df) * immigrant_fraction))
    immigrants = []
    for _ in range(k):
        p = generate_planar_3r_params()
        immigrants.append([p['link_lengths'], p['joint_limits']])

    imm_df = pd.DataFrame(immigrants, columns=children_df.columns)
    return pd.concat([children_df, imm_df], ignore_index=True)


# ============================================
# Connectivity evaluation (no elite-threshold logic)
# ============================================
def connectivity_analysis(df):
    connectivity = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):
        val = planar_3r_reliable_connectivity_analysis(row['link_lengths'], row['joint_limits'])
        #print(val)
        connectivity.append(val)

    df['connectivity'] = connectivity
    return df


# ============================================
# Elitism – clean & correct top-α selection
# ============================================
def elitism(df, alpha):
    sorted_df = df.sort_values("connectivity", ascending=False)
    elite_size = max(1, int(len(df) * alpha))

    elites = sorted_df.head(elite_size).iloc[:, :-1]
    return elites


# ============================================
# Next generation
# ============================================
def next_generation(df, mutation_rate):
    # 1) Evaluate connectivity
    df = connectivity_analysis(df)

    # 2) Select elites
    elites = elitism(df, alpha)

    print("\n=== Elite Group (Top {}%) ===".format(int(alpha * 100)))
    elite_view = df.sort_values("connectivity", ascending=False).head(len(elites))
    print(elite_view[['link_lengths', 'joint_limits', 'connectivity']])
    print("Best connectivity:", elite_view['connectivity'].max())
    print("Worst elite connectivity:", elite_view['connectivity'].min())
    print("======================================\n")

    # 3) Reproduce children
    num_children = len(df) - len(elites)
    children = []

    param_df = df.iloc[:, :-1]
    elite_list = elites.values.tolist()

    while len(children) < num_children:
        p1, p2 = stochastic_selection(param_df)
        c1, c2 = blx_alpha_crossover(p1, p2)

        c1_list = [c1['link_lengths'], c1['joint_limits']]
        c2_list = [c2['link_lengths'], c2['joint_limits']]

        if c1_list not in elite_list and c1_list not in children:
            children.append(c1_list)
        if len(children) < num_children:
            if c2_list not in elite_list and c2_list not in children:
                children.append(c2_list)

    children_df = pd.DataFrame(children, columns=param_df.columns)

    # 4) Mutate children
    children_df = apply_mutation(children_df, elites, mutation_rate)

    # 5) Inject random immigrants
    children_df = inject_immigrants(children_df)

    # 6) Rebuild next generation
    next_gen = pd.concat([elites, children_df], ignore_index=True)

    # 7) Adaptive mutation (mild)
    if df['connectivity'].max() == df['connectivity'].max():  # stagnation check placeholder
        mutation_rate = min(0.3, mutation_rate * 1.05)         # slowly increase
    else:
        mutation_rate = max(0.05, mutation_rate * 0.95)        # decrease when improving

    return next_gen.iloc[:sample_number], mutation_rate


# ============================================
# Save / Load
# ============================================
def save_population_to_txt(df, filename=POPULATION_FILE):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "link_lengths": list(map(float, row["link_lengths"])),
                "joint_limits": [list(map(float, jl)) for jl in row["joint_limits"]],
                "connectivity": float(row["connectivity"]) if "connectivity" in row else None
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Population saved.")


def load_population_from_txt(filename=POPULATION_FILE):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


# ============================================
# Main GA Loop
# ============================================
if __name__ == "__main__":

    try:
        df = load_population_from_txt(POPULATION_FILE)
        print("[INFO] Loaded saved population.")
        if "connectivity" in df.columns:
            df = df.drop(columns=["connectivity"])
    except FileNotFoundError:
        print("[INFO] No saved population; creating new.")
        df = pd.DataFrame([generate_planar_3r_params() for _ in range(sample_number)])

    rate = mutation_rate

    for gen in range(num_generations):
        print(f"\n=== Generation {gen} ===")
        df, rate = next_generation(df, rate)
        save_population_to_txt(df)
        df = load_population_from_txt()
        if "connectivity" in df.columns:
            df = df.drop(columns=["connectivity"])

