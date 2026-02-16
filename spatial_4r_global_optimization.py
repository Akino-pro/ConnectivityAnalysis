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
sample_number = 120
num_generations = 1  # adjust as you like

alpha_elite = 0.05             # top 5% elitism
blx_alpha = 0.3                # BLX-α crossover expansion
mutation_rate_init = 0.15      # initial mutation probability per generation
immigrant_fraction = 0.05      # random restart fraction
grid_sample_num = 162          # << you can tune this

POPULATION_FILE = "population_saved_spatial4R_global.txt"

# =========================================================
# Parameter constraints (now optimized)
# =========================================================
THETA_LO, THETA_HI = -np.pi, np.pi
ALPHA_LO, ALPHA_HI = -np.pi / 2, np.pi / 2
L_LO, L_HI = 0.0, 1.0
D_LO, D_HI = -1.0, 1.0


# =========================================================
# Helpers for hashing / duplicate detection
# (now includes CA + alpha + l + d)
# =========================================================
def ind_to_key(ind, ndigits=6):
    """
    Convert individual to a hashable rounded tuple.
    Includes: CA (8 vals) + alpha (4) + l (4) + d (4) => 20 numbers.
    """
    flat = []
    for lo, hi in ind["CA"]:
        flat.append(round(float(lo), ndigits))
        flat.append(round(float(hi), ndigits))
    for x in ind["alpha"]:
        flat.append(round(float(x), ndigits))
    for x in ind["l"]:
        flat.append(round(float(x), ndigits))
    for x in ind["d"]:
        flat.append(round(float(x), ndigits))
    return tuple(flat)


# =========================================================
# Individual generation
# CA format expected by your fitness:
#   CA = [(lo1,hi1), (lo2,hi2), (lo3,hi3), (lo4,hi4)]
# Now also optimize alpha, l, d with constraints:
#   -pi/2<=alpha_i<=pi/2, 0<=l_i<=1, -1<=d_i<=1
# Theta endpoints still in [-pi, pi]
# =========================================================
def generate_individual():
    CA = []
    for _ in range(4):
        lo = np.random.uniform(THETA_LO, THETA_HI)
        hi = np.random.uniform(THETA_LO, THETA_HI)
        CA.append((float(min(lo, hi)), float(max(lo, hi))))

    alpha = np.random.uniform(ALPHA_LO, ALPHA_HI, size=4).astype(float).tolist()
    l = np.random.uniform(L_LO, L_HI, size=4).astype(float).tolist()
    d = np.random.uniform(D_LO, D_HI, size=4).astype(float).tolist()

    return {"CA": CA, "alpha": alpha, "l": l, "d": d}


# =========================================================
# Chromosome <-> Individual
# Chromosome is 20 numbers:
#   [lo1,hi1, lo2,hi2, lo3,hi3, lo4,hi4, alpha1..4, l1..4, d1..4]
# =========================================================
def extract_chromosome(ind):
    CA = ind["CA"]
    chrom = []
    for (lo, hi) in CA:
        chrom.extend([lo, hi])
    chrom.extend(ind["alpha"])
    chrom.extend(ind["l"])
    chrom.extend(ind["d"])
    return np.array(chrom, dtype=float)


def enforce_constraints(chrom):
    chrom = np.array(chrom, dtype=float)
    if chrom.size != 20:
        raise ValueError(f"Chromosome must have length 20, got {chrom.size}")

    # --- theta ranges (CA) ---
    ca_raw = chrom[:8].reshape(4, 2)
    ca_raw = np.clip(ca_raw, THETA_LO, THETA_HI)
    lo = np.minimum(ca_raw[:, 0], ca_raw[:, 1])
    hi = np.maximum(ca_raw[:, 0], ca_raw[:, 1])
    CA = [(float(lo[i]), float(hi[i])) for i in range(4)]

    # --- alpha ---
    alpha = np.clip(chrom[8:12], ALPHA_LO, ALPHA_HI).astype(float).tolist()

    # --- l ---
    l = np.clip(chrom[12:16], L_LO, L_HI).astype(float).tolist()

    # --- d ---
    d = np.clip(chrom[16:20], D_LO, D_HI).astype(float).tolist()

    return {"CA": CA, "alpha": alpha, "l": l, "d": d}


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
        alpha = np.array(row["alpha"], dtype=float)
        l = np.array(row["l"], dtype=float)
        d = np.array(row["d"], dtype=float)

        val = spatial_4r_weighted_sum(
            grid_sample_num=grid_sample_num,
            d=d,
            alpha=alpha,
            l=l,
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
    return {
        "CA": winner["CA"],
        "alpha": winner["alpha"],
        "l": winner["l"],
        "d": winner["d"],
    }


# =========================================================
# Mutation: regenerate individual (fast + robust)
# =========================================================
def mutate_child():
    return generate_individual()


def apply_mutation(children_df, existing_keys, mutation_rate):
    """
    children_df contains columns: "CA", "alpha", "l", "d"
    existing_keys: set of keys already used (elites + already accepted children)
    """
    num_mutate = int(len(children_df) * mutation_rate)
    if num_mutate <= 0:
        return children_df

    mutated = children_df.copy()
    mutate_indices = mutated.sample(num_mutate).index

    for idx in mutate_indices:
        ind_new = mutate_child()
        key_new = ind_to_key(ind_new)
        # re-roll until unique
        tries = 0
        while key_new in existing_keys:
            ind_new = mutate_child()
            key_new = ind_to_key(ind_new)
            tries += 1
            if tries > 2000:
                # last-resort: accept to avoid infinite loop
                break

        mutated.at[idx, "CA"] = ind_new["CA"]
        mutated.at[idx, "alpha"] = ind_new["alpha"]
        mutated.at[idx, "l"] = ind_new["l"]
        mutated.at[idx, "d"] = ind_new["d"]
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
        key = ind_to_key(ind)
        tries = 0
        while key in existing_keys:
            ind = generate_individual()
            key = ind_to_key(ind)
            tries += 1
            if tries > 2000:
                break

        children_df.at[idx, "CA"] = ind["CA"]
        children_df.at[idx, "alpha"] = ind["alpha"]
        children_df.at[idx, "l"] = ind["l"]
        children_df.at[idx, "d"] = ind["d"]
        existing_keys.add(key)

    return children_df


# =========================================================
# Save / Load (save params only; no fitness)
# =========================================================
def save_population_to_txt(df_params_only, filename=POPULATION_FILE):
    with open(filename, "w") as f:
        for _, row in df_params_only.iterrows():
            entry = {
                "CA": [[float(a), float(b)] for (a, b) in row["CA"]],
                "alpha": [float(x) for x in row["alpha"]],
                "l": [float(x) for x in row["l"]],
                "d": [float(x) for x in row["d"]],
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Population saved to: {filename}")


def load_population_from_txt(filename=POPULATION_FILE):
    data = []
    with open(filename, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            CA = [(float(pair[0]), float(pair[1])) for pair in d["CA"]]
            alpha = [float(x) for x in d["alpha"]]
            l = [float(x) for x in d["l"]]
            dd = [float(x) for x in d["d"]]
            data.append({"CA": CA, "alpha": alpha, "l": l, "d": dd})
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
    print(elites_fit[["CA", "alpha", "l", "d", "fitness"]])
    print("Best fitness:", best_now)
    print("Worst elite fitness:", worst_elite)
    print("======================================\n")

    # 3) Reproduce children to fill remaining slots
    num_children = sample_number - elite_size
    children = []

    # Existing keys: elites + accepted children
    existing_keys = set(
        ind_to_key({"CA": r["CA"], "alpha": r["alpha"], "l": r["l"], "d": r["d"]})
        for _, r in elites_fit.iterrows()
    )

    # Tournament selection uses fitness, so use df_fit
    while len(children) < num_children:
        p1 = tournament_select(df_fit, k=3)
        p2 = tournament_select(df_fit, k=3)
        c1, c2 = blx_alpha_crossover(p1, p2)

        for c in (c1, c2):
            if len(children) >= num_children:
                break
            key = ind_to_key(c)
            if key not in existing_keys:
                children.append({"CA": c["CA"], "alpha": c["alpha"], "l": c["l"], "d": c["d"]})
                existing_keys.add(key)

    children_df = pd.DataFrame(children, columns=["CA", "alpha", "l", "d"])

    # 4) Mutate some children (keeps uniqueness)
    children_df = apply_mutation(children_df, existing_keys, mutation_rate)

    # 5) Replace some children with immigrants (keeps size stable)
    children_df = inject_immigrants_inplace(children_df, existing_keys, immigrant_fraction)

    # 6) Next population (params only)
    elites_params = elites_fit[["CA", "alpha", "l", "d"]].copy()
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
