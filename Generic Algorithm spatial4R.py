import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# IMPORT YOUR FITNESS FUNCTION HERE
# =========================================================
from spatial4R_reliable_connectivity_analysis import spatial_4r_weighted_sum

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

# =========================================================
# GA Hyperparameters
# =========================================================
sample_number = 48
num_generations = 2  # adjust as you like

alpha_elite = 0.05             # top 5% elitism
blx_alpha = 0.3                # BLX-α crossover expansion
mutation_rate_init = 0.15      # initial mutation probability per generation
immigrant_fraction = 0.05      # random restart fraction
grid_sample_num = 162          # << you can tune this

POPULATION_FILE = "population_saved_spatial4R_thetaOnly.txt"

# =========================================================
# Fixed DH parameters (constant across population)
# =========================================================
DH_ALPHA = np.array([85, -53, -89, 68]) * np.pi / 180.0
L_CONST = np.array([0.50, 0.48, 0.76, 0.95])  # 'l' in your fitness signature
D_CONST = np.array([-0.29, 0.00, 0.05, 1.00])


# =========================================================
# Helpers for hashing / duplicate detection
# =========================================================
def ca_to_key(CA, ndigits=8):
    """Convert CA = [(lo,hi),...]*4 to a hashable rounded tuple."""
    flat = []
    for lo, hi in CA:
        flat.append(round(float(lo), ndigits))
        flat.append(round(float(hi), ndigits))
    return tuple(flat)


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
    chrom = np.clip(chrom, -np.pi, np.pi)
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
# Fitness evaluation
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

    out = df.copy()
    out["fitness"] = fits
    return out


# =========================================================
# Elitism
# =========================================================
def get_elites(df_with_fit, alpha):
    sorted_df = df_with_fit.sort_values("fitness", ascending=False).reset_index(drop=True)
    elite_size = max(1, int(len(sorted_df) * alpha))
    elites = sorted_df.head(elite_size).copy()
    return elites


# =========================================================
# Selection: tournament selection (adds selection pressure)
# =========================================================
def tournament_select(df_with_fit, k=3):
    """Pick the best among k random individuals."""
    sample = df_with_fit.sample(n=min(k, len(df_with_fit)))
    winner = sample.sort_values("fitness", ascending=False).iloc[0]
    return {"CA": winner["CA"]}


# =========================================================
# Mutation: regenerate CA (fast + robust)
# =========================================================
def mutate_child():
    return generate_individual()["CA"]


def apply_mutation(children_df, existing_keys, mutation_rate):
    """
    children_df contains column "CA".
    existing_keys: set of keys already used (elites + already accepted children)
    """
    num_mutate = int(len(children_df) * mutation_rate)
    if num_mutate <= 0:
        return children_df

    mutated = children_df.copy()
    mutate_indices = mutated.sample(num_mutate).index

    for idx in mutate_indices:
        CA_new = mutate_child()
        key_new = ca_to_key(CA_new)
        # re-roll until unique
        tries = 0
        while key_new in existing_keys:
            CA_new = mutate_child()
            key_new = ca_to_key(CA_new)
            tries += 1
            if tries > 2000:
                # last-resort: accept to avoid infinite loop
                break
        mutated.at[idx, "CA"] = CA_new
        existing_keys.add(key_new)

    return mutated


# =========================================================
# Inject immigrants by replacing some children (keeps size stable)
# =========================================================
def inject_immigrants_inplace(children_df, existing_keys, frac):
    k = max(1, int(len(children_df) * frac))
    replace_idx = children_df.sample(k).index

    for idx in replace_idx:
        ind = generate_individual()
        key = ca_to_key(ind["CA"])
        tries = 0
        while key in existing_keys:
            ind = generate_individual()
            key = ca_to_key(ind["CA"])
            tries += 1
            if tries > 2000:
                break
        children_df.at[idx, "CA"] = ind["CA"]
        existing_keys.add(key)

    return children_df


# =========================================================
# Save / Load (save ONLY CA; no fitness)
# =========================================================
def save_population_to_txt(df_params_only, filename=POPULATION_FILE):
    with open(filename, "w") as f:
        for _, row in df_params_only.iterrows():
            entry = {"CA": [[float(a), float(b)] for (a, b) in row["CA"]]}
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Population saved to: {filename}")


def load_population_from_txt(filename=POPULATION_FILE):
    data = []
    with open(filename, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            CA = [(float(pair[0]), float(pair[1])) for pair in d["CA"]]
            data.append({"CA": CA})
    return pd.DataFrame(data)


# =========================================================
# Next generation step
# =========================================================
def next_generation(df_params_only, mutation_rate, best_so_far):
    # 1) Evaluate fitness
    df_fit = fitness_analysis(df_params_only)

    # 2) Elites
    elites_fit = get_elites(df_fit, alpha_elite)
    elite_size = len(elites_fit)

    best_now = float(df_fit["fitness"].max())
    worst_elite = float(elites_fit["fitness"].min())

    print("\n=== Elite Group (Top {}%) ===".format(int(alpha_elite * 100)))
    print(elites_fit[["CA", "fitness"]])
    print("Best fitness:", best_now)
    print("Worst elite fitness:", worst_elite)
    print("======================================\n")

    # 3) Reproduce children to fill remaining slots
    num_children = sample_number - elite_size
    children = []

    # Existing keys: elites + accepted children
    existing_keys = set(ca_to_key(CA) for CA in elites_fit["CA"].tolist())

    # Tournament selection uses fitness, so use df_fit
    while len(children) < num_children:
        p1 = tournament_select(df_fit, k=3)
        p2 = tournament_select(df_fit, k=3)
        c1, c2 = blx_alpha_crossover(p1, p2)

        for c in (c1, c2):
            if len(children) >= num_children:
                break
            key = ca_to_key(c["CA"])
            if key not in existing_keys:
                children.append({"CA": c["CA"]})
                existing_keys.add(key)

    children_df = pd.DataFrame(children)

    # 4) Mutate some children (keeps uniqueness)
    children_df = apply_mutation(children_df, existing_keys, mutation_rate)

    # 5) Replace some children with immigrants (keeps size stable)
    children_df = inject_immigrants_inplace(children_df, existing_keys, immigrant_fraction)

    # 6) Next population (params only)
    elites_params = elites_fit[["CA"]].copy()
    next_pop = pd.concat([elites_params, children_df], ignore_index=True)

    # Safety trim (should already be exact)
    next_pop = next_pop.iloc[:sample_number].reset_index(drop=True)

    # 7) Adaptive mutation
    improved = (best_now > best_so_far + 1e-12)
    if not improved:
        mutation_rate = min(0.35, mutation_rate * 1.08)
    else:
        mutation_rate = max(0.05, mutation_rate * 0.95)
        best_so_far = best_now

    return next_pop, mutation_rate, best_so_far


# =========================================================
# Main GA Loop
# =========================================================
if __name__ == "__main__":
    # Load or initialize population (params only)
    if os.path.exists(POPULATION_FILE):
        df = load_population_from_txt(POPULATION_FILE)
        print("[INFO] Loaded saved population.")
        # If file has wrong size, fix it
        if len(df) < sample_number:
            extra = [generate_individual() for _ in range(sample_number - len(df))]
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        elif len(df) > sample_number:
            df = df.iloc[:sample_number].reset_index(drop=True)
    else:
        print("[INFO] No saved population; creating new.")
        df = pd.DataFrame([generate_individual() for _ in range(sample_number)])

    rate = mutation_rate_init
    best_so_far = -np.inf

    for gen in range(num_generations):
        print(f"\n=== Generation {gen} ===")
        df, rate, best_so_far = next_generation(df, rate, best_so_far)

        # IMPORTANT: save immediately after generation completes (so you can stop anytime)
        save_population_to_txt(df, POPULATION_FILE)

    print("[INFO] GA finished.")
