import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from planar3R_reliable_connectivity_analysis import planar_3r_reliable_connectivity_analysis

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

# ============================================
# GA Hyperparameters
# ============================================
sample_number = 128
num_generations = 1

elite_fraction = 0.05          # keep top fraction as elites
blx_alpha = 0.30               # BLX-α crossover parameter

mutation_rate_init = 0.15      # will adapt
immigrant_fraction = 0.05      # random immigrants fraction
POPULATION_FILE = "population_saved_planar_3R_global.txt"

# ============================================
# Utilities: hashing individuals (robust uniqueness)
# ============================================
def indiv_key(link_lengths, joint_limits, nd=6):
    ll = tuple(round(float(x), nd) for x in link_lengths)
    jl = tuple((round(float(a), nd), round(float(b), nd)) for a, b in joint_limits)
    return (ll, jl)

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

    return {"link_lengths": link_lengths, "joint_limits": joint_limits}

# ============================================
# Chromosome extraction / enforcing constraints
# ============================================
def extract_chromosome(parent):
    l1, l2, _ = parent["link_lengths"]

    # Joint 1 is symmetric: [-u1, u1]
    _, jl1_up = parent["joint_limits"][0]
    u1 = jl1_up

    jl2_low, jl2_up = parent["joint_limits"][1]
    jl3_low, jl3_up = parent["joint_limits"][2]

    return np.array(
        [l1, l2, u1,
         jl2_low, jl2_up,
         jl3_low, jl3_up],
        dtype=float
    )

def enforce_constraints(chrom):
    # lengths: l1,l2 free; l3 determined by sum=3
    l1 = max(float(chrom[0]), 1e-6)
    l2 = max(float(chrom[1]), 1e-6)
    l3 = 3.0 - (l1 + l2)

    # if l3 invalid, renormalize l1,l2 then recompute l3
    if l3 <= 0:
        total = l1 + l2
        l1 = 3.0 * (l1 / total)
        l2 = 3.0 * (l2 / total)
        l3 = 3.0 - (l1 + l2)

    # Joint 1 symmetric range
    u1 = float(np.clip(chrom[2], 0.01, np.pi))
    jl1_low, jl1_up = -u1, u1

    # Joint 2
    jl2_low = float(np.clip(chrom[3], -np.pi, np.pi))
    jl2_up  = float(np.clip(chrom[4], -np.pi, np.pi))
    if jl2_low > jl2_up:
        jl2_low, jl2_up = jl2_up, jl2_low

    # Joint 3
    jl3_low = float(np.clip(chrom[5], -np.pi, np.pi))
    jl3_up  = float(np.clip(chrom[6], -np.pi, np.pi))
    if jl3_low > jl3_up:
        jl3_low, jl3_up = jl3_up, jl3_low

    # Avoid degenerate intervals
    eps = 1e-3
    if jl2_up - jl2_low < eps:
        jl2_up = min(np.pi, jl2_low + eps)
    if jl3_up - jl3_low < eps:
        jl3_up = min(np.pi, jl3_low + eps)
    if jl1_up - jl1_low < eps:
        jl1_up = min(np.pi, jl1_low + eps)

    return {
        "link_lengths": [l1, l2, l3],
        "joint_limits": [(jl1_low, jl1_up), (jl2_low, jl2_up), (jl3_low, jl3_up)],
    }

# ============================================
# BLX-α crossover
# ============================================
def blx_alpha_crossover(parent1, parent2, alpha=0.3):
    p1 = extract_chromosome(parent1)
    p2 = extract_chromosome(parent2)

    child1 = np.zeros_like(p1, dtype=float)
    child2 = np.zeros_like(p1, dtype=float)

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
# Selection: tournament (fitness-aware)
# ============================================
def tournament_selection(df_with_fit, k=3):
    k = min(k, len(df_with_fit))
    cand = df_with_fit.sample(n=k, replace=False)
    return cand.sort_values("connectivity", ascending=False).iloc[0].to_dict()

# ============================================
# Mutation: full resample replacement
# ============================================
def mutate_child():
    p = generate_planar_3r_params()
    return [p["link_lengths"], p["joint_limits"]]

def apply_mutation(children_df, used_keys, mutation_rate):
    n = len(children_df)
    num_mutate = int(n * mutation_rate)
    if num_mutate <= 0:
        return children_df

    mutated = children_df.copy()
    mutate_indices = mutated.sample(n=num_mutate, replace=False).index.tolist()

    for idx in mutate_indices:
        tries = 0
        while True:
            tries += 1
            ll, jl = mutate_child()
            k = indiv_key(ll, jl)
            if k not in used_keys:
                mutated.at[idx, "link_lengths"] = ll
                mutated.at[idx, "joint_limits"] = jl
                used_keys.add(k)
                break
            if tries > 200:
                # fallback: accept even if duplicate (rare)
                mutated.at[idx, "link_lengths"] = ll
                mutated.at[idx, "joint_limits"] = jl
                used_keys.add(k)
                break

    return mutated

# ============================================
# Immigrants: inject, then sample back down
# ============================================
def inject_immigrants(children_df, num_children_target, used_keys, immigrant_fraction):
    k = max(1, int(num_children_target * immigrant_fraction))
    immigrants = []
    tries_total = 0

    while len(immigrants) < k and tries_total < 5000:
        tries_total += 1
        p = generate_planar_3r_params()
        ll, jl = p["link_lengths"], p["joint_limits"]
        key = indiv_key(ll, jl)
        if key in used_keys:
            continue
        immigrants.append([ll, jl])
        used_keys.add(key)

    if immigrants:
        imm_df = pd.DataFrame(immigrants, columns=children_df.columns)
        children_df = pd.concat([children_df, imm_df], ignore_index=True)

    # sample back to target size to avoid truncation bias
    if len(children_df) > num_children_target:
        children_df = children_df.sample(n=num_children_target, replace=False).reset_index(drop=True)
    else:
        children_df = children_df.reset_index(drop=True)

    return children_df

# ============================================
# Fitness evaluation (sequential)
# ============================================
def connectivity_analysis(df):
    connectivity = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):
        val = planar_3r_reliable_connectivity_analysis(row["link_lengths"], row["joint_limits"])
        connectivity.append(float(val))
    out = df.copy()
    out["connectivity"] = connectivity
    return out

# ============================================
# Elitism
# ============================================
def elitism(df_with_fit, elite_fraction):
    sorted_df = df_with_fit.sort_values("connectivity", ascending=False)
    elite_size = max(1, int(len(df_with_fit) * elite_fraction))
    elites = sorted_df.head(elite_size).drop(columns=["connectivity"]).reset_index(drop=True)
    return elites

# ============================================
# Next generation
# ============================================
def next_generation(df, mutation_rate, prev_best=None):
    # 1) Evaluate fitness of CURRENT generation (needed for selection)
    df_fit = connectivity_analysis(df)
    curr_best = float(df_fit["connectivity"].max())

    # 2) Select elites
    elites = elitism(df_fit, elite_fraction)
    elite_size = len(elites)

    # Print elite stats
    elite_view = df_fit.sort_values("connectivity", ascending=False).head(elite_size)
    print(f"\n=== Elite Group (Top {int(elite_fraction * 100)}%) ===")
    print(elite_view[["link_lengths", "joint_limits", "connectivity"]])
    print("Best connectivity:", float(elite_view["connectivity"].max()))
    print("Worst elite connectivity:", float(elite_view["connectivity"].min()))
    print("======================================\n")

    # 3) Reproduce children (tournament + BLX-α) with uniqueness + fallback
    num_children = len(df) - elite_size

    elites_keys = set(indiv_key(ll, jl) for ll, jl in elites[["link_lengths", "joint_limits"]].values.tolist())
    used_keys = set(elites_keys)

    children_rows = []
    max_tries = 50 * max(1, num_children)
    tries = 0

    while len(children_rows) < num_children and tries < max_tries:
        tries += 1
        p1 = tournament_selection(df_fit, k=3)
        p2 = tournament_selection(df_fit, k=3)
        c1, c2 = blx_alpha_crossover(p1, p2, alpha=blx_alpha)

        for c in (c1, c2):
            if len(children_rows) >= num_children:
                break
            ll, jl = c["link_lengths"], c["joint_limits"]
            k = indiv_key(ll, jl)
            if k in used_keys:
                continue
            children_rows.append([ll, jl])
            used_keys.add(k)

    # Fallback: fill remaining with random individuals
    while len(children_rows) < num_children:
        p = generate_planar_3r_params()
        ll, jl = p["link_lengths"], p["joint_limits"]
        k = indiv_key(ll, jl)
        if k in used_keys:
            continue
        children_rows.append([ll, jl])
        used_keys.add(k)

    children_df = pd.DataFrame(children_rows, columns=["link_lengths", "joint_limits"])

    # 4) Mutate children (replacement-based), respecting uniqueness
    children_df = apply_mutation(children_df, used_keys, mutation_rate)

    # 5) Inject immigrants, then sample back down to exactly num_children
    children_df = inject_immigrants(children_df, num_children, used_keys, immigrant_fraction)

    # 6) Build next generation (PARAMS ONLY; no connectivity stored)
    next_gen = pd.concat([elites, children_df], ignore_index=True)
    next_gen = next_gen.sample(n=sample_number, replace=False).reset_index(drop=True)

    # 7) Adaptive mutation based on improvement vs prev_best
    if prev_best is None or curr_best <= prev_best + 1e-12:
        mutation_rate = min(0.30, mutation_rate * 1.05)
        stagnated = True
    else:
        mutation_rate = max(0.05, mutation_rate * 0.95)
        stagnated = False

    return next_gen, mutation_rate, curr_best, stagnated

# ============================================
# Save / Load (JSONL) -- NO CONNECTIVITY SAVED
# ============================================
def save_population_to_txt(df, filename=POPULATION_FILE):
    """
    Save ONLY parameters needed to resume GA later.
    connectivity is intentionally NOT saved.
    """
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "link_lengths": list(map(float, row["link_lengths"])),
                "joint_limits": [list(map(float, jl)) for jl in row["joint_limits"]],
            }
            f.write(json.dumps(entry) + "\n")
    print("[INFO] Population saved (params only).")

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
    # Load or init population
    try:
        df = load_population_from_txt(POPULATION_FILE)
        print("[INFO] Loaded saved population.")
    except FileNotFoundError:
        print("[INFO] No saved population; creating new.")
        df = pd.DataFrame([generate_planar_3r_params() for _ in range(sample_number)])

    rate = mutation_rate_init
    prev_best = None

    for gen in range(num_generations):
        print(f"\n=== Generation {gen} ===")
        df, rate, prev_best, stagnated = next_generation(df, rate, prev_best)
        print(f"[INFO] mutation_rate={rate:.4f}  stagnated={stagnated}  best={prev_best:.6f}")

        # Save immediately after new generation is generated (params only)
        save_population_to_txt(df, POPULATION_FILE)

        # Optional: reload to mimic "resume later" behavior (keeps types consistent)
        df = load_population_from_txt(POPULATION_FILE)
