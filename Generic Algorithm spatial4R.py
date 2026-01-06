import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# IMPORT YOUR FITNESS FUNCTION HERE
# =========================================================
# Make sure this import path matches your project structure.
from spatial4R_reliable_connectivity_analysis import spatial_4r_weighted_sum


pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


# =========================================================
# GA Hyperparameters
# =========================================================
sample_number = 128
num_generations = 1

alpha_elite = 0.05             # top 5% elitism (NOT DH alpha)
blx_alpha = 0.3                # BLX-α crossover expansion
mutation_rate_init = 0.15      # mutation probability per generation
immigrant_fraction = 0.05      # random restart fraction
grid_sample_num = 162#128 200 288        # << you can tune this

POPULATION_FILE = "population_saved_spatial4R_thetaOnly.txt"


# =========================================================
# Fixed DH parameters (constant across population)
# NOTE: you said GA only optimizes theta ranges, so alpha/d/l fixed
# =========================================================
DH_ALPHA = np.array([85, -53, -89, 68]) * np.pi / 180.0
L_CONST  = np.array([0.50, 0.48, 0.76, 0.95])  # 'l' in your fitness signature
D_CONST  = np.array([-0.29, 0.00, 0.05, 1.00])


# =========================================================
# Individual generation: theta ranges only
# CA format expected by your fitness:
#   CA = [(lo1,hi1), (lo2,hi2), (lo3,hi3), (lo4,hi4)]
# =========================================================
def generate_individual():
    CA = []
    for _ in range(4):
        lo = np.random.uniform(-np.pi, np.pi)
        hi = np.random.uniform(-np.pi, np.pi)
        CA.append((float(min(lo, hi)), float(max(lo, hi))))
    return {"CA": CA}


# =========================================================
# Chromosome <-> Individual
# Chromosome is 8 numbers: [lo1,hi1, lo2,hi2, lo3,hi3, lo4,hi4]
# =========================================================
def extract_chromosome(ind):
    CA = ind["CA"]
    chrom = []
    for (lo, hi) in CA:
        chrom.extend([lo, hi])
    return np.array(chrom, dtype=float)


def enforce_constraints(chrom):
    chrom = np.array(chrom, dtype=float).reshape(4, 2)

    # clamp all bounds to [-pi, pi]
    chrom = np.clip(chrom, -np.pi, np.pi)

    # ensure lo <= hi per joint
    lo = np.minimum(chrom[:, 0], chrom[:, 1])
    hi = np.maximum(chrom[:, 0], chrom[:, 1])

    CA = [(float(lo[i]), float(hi[i])) for i in range(4)]
    return {"CA": CA}


# =========================================================
# BLX-α crossover
# =========================================================
def blx_alpha_crossover(parent1, parent2, alpha=blx_alpha):
    p1 = extract_chromosome(parent1)
    p2 = extract_chromosome(parent2)

    child1 = np.zeros_like(p1)
    child2 = np.zeros_like(p2)

    for i in range(len(p1)):
        mn = min(p1[i], p2[i])
        mx = max(p1[i], p2[i])
        diff = mx - mn

        low = mn - alpha * diff
        high = mx + alpha * diff

        child1[i] = np.random.uniform(low, high)
        child2[i] = np.random.uniform(low, high)

    return enforce_constraints(child1), enforce_constraints(child2)


# =========================================================
# Parent selection (simple stochastic)
# =========================================================
def stochastic_selection(df_params_only):
    p1 = df_params_only.sample().iloc[0].to_dict()
    p2 = df_params_only.sample().iloc[0].to_dict()
    return p1, p2


# =========================================================
# Mutation: replace CA entirely (fast + robust)
# Later you can switch to Gaussian perturbation if desired.
# =========================================================
def mutate_child():
    return [generate_individual()["CA"]]


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
        mutated.at[idx, "CA"] = c[0]
        pop_list = mutated.values.tolist()

    return mutated


# =========================================================
# Random immigrants
# =========================================================
def inject_immigrants(children_df):
    k = max(1, int(len(children_df) * immigrant_fraction))
    immigrants = [{"CA": generate_individual()["CA"]} for _ in range(k)]
    imm_df = pd.DataFrame(immigrants, columns=children_df.columns)
    return pd.concat([children_df, imm_df], ignore_index=True)


# =========================================================
# Fitness evaluation
# fitness = spatial_4r_weighted_sum(grid_sample_num, d, alpha, l, CA)
# =========================================================
def fitness_analysis(df):
    fits = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):
        CA = row["CA"]
        val = spatial_4r_weighted_sum(
            grid_sample_num=grid_sample_num,
            d=D_CONST,
            alpha=DH_ALPHA,
            l=L_CONST,
            CA=CA
        )
        fits.append(float(val))

    df["fitness"] = fits
    return df


# =========================================================
# Elitism
# =========================================================
def elitism(df, alpha):
    sorted_df = df.sort_values("fitness", ascending=False)
    elite_size = max(1, int(len(df) * alpha))
    elites = sorted_df.head(elite_size).iloc[:, :-1]  # drop fitness column
    return elites


# =========================================================
# Next generation
# =========================================================
def next_generation(df, mutation_rate, best_so_far):
    # 1) Evaluate fitness
    df = fitness_analysis(df)

    # 2) Select elites
    elites = elitism(df, alpha_elite)

    elite_view = df.sort_values("fitness", ascending=False).head(len(elites))
    best_now = float(df["fitness"].max())
    worst_elite = float(elite_view["fitness"].min())

    print("\n=== Elite Group (Top {}%) ===".format(int(alpha_elite * 100)))
    print(elite_view[["CA", "fitness"]])
    print("Best fitness:", best_now)
    print("Worst elite fitness:", worst_elite)
    print("======================================\n")

    # 3) Reproduce children
    num_children = len(df) - len(elites)
    children = []

    param_df = df.iloc[:, :-1]       # drop fitness
    elite_list = elites.values.tolist()

    while len(children) < num_children:
        p1, p2 = stochastic_selection(param_df)
        c1, c2 = blx_alpha_crossover(p1, p2)

        c1_list = [c1["CA"]]
        c2_list = [c2["CA"]]

        if c1_list not in elite_list and c1_list not in children:
            children.append(c1_list)
        if len(children) < num_children:
            if c2_list not in elite_list and c2_list not in children:
                children.append(c2_list)

    children_df = pd.DataFrame(children, columns=["CA"])

    # 4) Mutate children
    children_df = apply_mutation(children_df, elites, mutation_rate)

    # 5) Inject random immigrants
    children_df = inject_immigrants(children_df)

    # 6) Rebuild next generation (trim to sample_number later)
    next_gen = pd.concat([elites, children_df], ignore_index=True)

    # 7) Adaptive mutation: increase if stagnating, decrease if improving
    improved = (best_now > best_so_far + 1e-12)
    if not improved:
        mutation_rate = min(0.35, mutation_rate * 1.08)
    else:
        mutation_rate = max(0.05, mutation_rate * 0.95)
        best_so_far = best_now

    return next_gen.iloc[:sample_number], mutation_rate, best_so_far


# =========================================================
# Save / Load
# =========================================================
def save_population_to_txt(df, filename=POPULATION_FILE):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "CA": [[float(x[0]), float(x[1])] for x in row["CA"]],
                "fitness": float(row["fitness"]) if "fitness" in row and row["fitness"] is not None else None
            }
            f.write(json.dumps(entry) + "\n")
    print("[INFO] Population saved.")


def load_population_from_txt(filename=POPULATION_FILE):
    data = []
    with open(filename, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            CA = [(float(pair[0]), float(pair[1])) for pair in d["CA"]]
            data.append({"CA": CA, "fitness": d.get("fitness", None)})
    return pd.DataFrame(data)


# =========================================================
# Main GA Loop
# =========================================================
if __name__ == "__main__":
    try:
        df = load_population_from_txt(POPULATION_FILE)
        print("[INFO] Loaded saved population.")
        if "fitness" in df.columns:
            df = df.drop(columns=["fitness"])
    except FileNotFoundError:
        print("[INFO] No saved population; creating new.")
        df = pd.DataFrame([generate_individual() for _ in range(sample_number)])

    rate = mutation_rate_init
    best_so_far = -np.inf

    for gen in range(num_generations):
        print(f"\n=== Generation {gen} ===")
        df, rate, best_so_far = next_generation(df, rate, best_so_far)
        save_population_to_txt(df)
        df = load_population_from_txt()
        if "fitness" in df.columns:
            df = df.drop(columns=["fitness"])
