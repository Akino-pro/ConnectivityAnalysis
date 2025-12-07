import cv2
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import matplotlib

from greyscale_experiment import compute_connectivity

matplotlib.use("Agg")   # Non-GUI backend to reduce RAM
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from planar_3R_greyscale_reliability_map import (
    planar_3R_greyscale_connectivity_analysis,  # Overridden below
    compute_beta_range,
    union_ranges, # We override this with optimized version
    points, indices, color_list, ring_width
)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)


# ============================================================
# GA Hyperparameters
# ============================================================
sample_number = 128
num_generations = 20
alpha = 0.05
mutation_rate = 0.15


# ============================================================
# OPTIMIZED FIGURE-TO-GRAY (Option B)
# ============================================================
def figure_to_gray_image(fig):
    """
    Render a Matplotlib figure to grayscale image.
    Uses buffer_rgba() → ALWAYS available on FigureCanvasAgg.
    """

    # Force render
    fig.canvas.draw()

    # Direct RGBA array from canvas
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (H, W, 4)

    # Convert RGBA → RGB
    img_rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)

    # Convert RGB → grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Cleanup
    del buf, img_rgb
    gc.collect()

    return img_gray.astype(np.float32)


# ============================================================
# RAM-Optimized planar_3R_greyscale_connectivity_analysis
# ============================================================
def planar_3R_greyscale_connectivity_analysis(L, CA):
    """
    RAM-safe version — identical output.
    """
    fig3, ax2d3 = plt.subplots(figsize=(6, 6))

    s = np.sum(L)
    ax2d3.set_xlim(-s, s)
    ax2d3.set_ylim(-s, s)
    ax2d3.set_aspect('equal')
    ax2d3.axis('off')

    total_area = 0.0
    wedge_refs = []

    for i in range(len(points)):
        x, y = points[i]

        beta_ranges, reliable_beta_ranges = compute_beta_range(x, y, L, CA)
        ranges_to_compute = []

        for b_i in indices:
            b_r = reliable_beta_ranges[b_i]
            color = color_list[b_i]

            for beta_range in b_r:
                ranges_to_compute.append(beta_range)

                theta1 = np.degrees(beta_range[0])
                theta2 = np.degrees(beta_range[1])
                outer_r = x + ring_width/2.0

                wedge = Wedge(
                    center=(0, 0),
                    r=outer_r,
                    theta1=theta1,
                    theta2=theta2,
                    width=ring_width,
                    facecolor=color,
                    edgecolor=color,
                    alpha=1.0,
                    zorder=b_i + 1
                )
                ax2d3.add_patch(wedge)
                wedge_refs.append(wedge)

        if ranges_to_compute:
            merged = union_ranges(ranges_to_compute)
            inner = x - ring_width/2.0
            outer = x + ring_width/2.0
            full_area = np.pi * (outer**2 - inner**2)

            for br in merged:
                beta_start, beta_end = br
                frac = (beta_end - beta_start) / (2*np.pi)
                total_area += full_area * frac

        del beta_ranges, reliable_beta_ranges, ranges_to_compute
        gc.collect()

    img_gray = figure_to_gray_image(fig3)

    connectivity_value = compute_connectivity(img_gray)

    for w in wedge_refs:
        w.remove()
    del wedge_refs

    ax2d3.cla()
    plt.close(fig3)
    gc.collect()

    return connectivity_value


# ============================================================
# Parameter generation
# ============================================================
def generate_planar_3r_params():
    raw = np.random.dirichlet([1, 1, 1])
    link_lengths = (3 * raw).tolist()

    joint_limits = []
    for i in range(3):
        if i == 0:
            u = np.random.uniform(0, np.pi)
            joint_limits.append((-u, u))
        else:
            lo = np.random.uniform(-np.pi, np.pi)
            hi = np.random.uniform(lo, np.pi)
            joint_limits.append((lo, hi))

    return {
        'link_lengths': link_lengths,
        'joint_limits': joint_limits
    }


# ============================================================
# GA Operators
# ============================================================
def stochastic_selection(samples: pd.DataFrame):
    p1 = samples.sample().iloc[0].to_dict()
    p2 = samples.sample().iloc[0].to_dict()
    return p1, p2


def two_point_crossover(p1, p2):
    return (
        {
            'link_lengths': p1['link_lengths'],
            'joint_limits': p2['joint_limits']
        },
        {
            'link_lengths': p2['link_lengths'],
            'joint_limits': p1['joint_limits']
        }
    )


def mutate_child():
    params = generate_planar_3r_params()
    return [params['link_lengths'], params['joint_limits']]


def elitism(samples: pd.DataFrame, alpha=alpha):
    sorted_samples = samples.sort_values(by="connectivity", ascending=False)
    cutoff = max(1, int(len(sorted_samples) * alpha))

    elites = sorted_samples.head(cutoff).iloc[:, :-1]
    remaining = len(samples) - cutoff

    elite_set = {
        (tuple(r['link_lengths']), tuple(r['joint_limits']))
        for _, r in elites.iterrows()
    }

    children = []
    while len(children) < remaining:
        p1, p2 = stochastic_selection(samples.iloc[:, :-1])
        c1, c2 = two_point_crossover(p1, p2)

        cand1 = [c1['link_lengths'], c1['joint_limits']]
        cand2 = [c2['link_lengths'], c2['joint_limits']]

        if (tuple(cand1[0]), tuple(cand1[1])) not in elite_set:
            children.append(cand1)

        if len(children) < remaining:
            if (tuple(cand2[0]), tuple(cand2[1])) not in elite_set:
                children.append(cand2)

    new_children = pd.DataFrame.from_records(
        [{'link_lengths': ll, 'joint_limits': jl} for ll, jl in children]
    )

    return elites, new_children


def apply_mutation(pop: pd.DataFrame, elites: pd.DataFrame, mutation_rate=mutation_rate):

    n = int(len(pop) * mutation_rate)
    if n == 0:
        return pop

    mutated = pop.copy()

    pop_set = {
        (tuple(r['link_lengths']), tuple(r['joint_limits']))
        for _, r in mutated.iterrows()
    }
    elite_set = {
        (tuple(r['link_lengths']), tuple(r['joint_limits']))
        for _, r in elites.iterrows()
    }

    idxs = mutated.sample(n).index
    for idx in idxs:
        cand = mutate_child()

        while (tuple(cand[0]), tuple(cand[1])) in pop_set \
           or (tuple(cand[0]), tuple(cand[1])) in elite_set:
            cand = mutate_child()

        mutated.at[idx, 'link_lengths'] = cand[0]
        mutated.at[idx, 'joint_limits'] = cand[1]
        pop_set.add((tuple(cand[0]), tuple(cand[1])))

    return mutated


# ============================================================
# Connectivity evaluation
# ============================================================
def connectivity_analysis(samples: pd.DataFrame, global_elite_limit):
    vals = []
    elites = [global_elite_limit]
    limit_size = max(1, int(sample_number * alpha))

    for idx, row in tqdm(samples.iterrows(), total=len(samples),
                         desc="Evaluating samples"):

        v = planar_3R_greyscale_connectivity_analysis(
            row['link_lengths'], row['joint_limits']
        )
        vals.append(v)

        if v > elites[-1]:
            if len(elites) < limit_size:
                elites.append(v)
            else:
                elites[-1] = v
            elites = sorted(elites, reverse=True)

        del row
        gc.collect()

    samples["connectivity"] = vals

    sorted_samples = samples.sort_values(by="connectivity", ascending=False)
    cutoff = max(1, int(len(sorted_samples) * alpha))
    champion = sorted_samples.head(cutoff)

    new_limit = max(global_elite_limit, champion.iloc[-1]['connectivity'])

    print("Current elite band (top {}%):".format(int(alpha * 100)))
    print(champion)

    return samples, new_limit


def next_generation(samples, global_elite_limit, alpha=alpha):
    samples, global_elite_limit = connectivity_analysis(samples, global_elite_limit)

    elites, children = elitism(samples, alpha)
    children = apply_mutation(children, elites)

    next_gen = pd.concat([elites, children], ignore_index=True)

    return next_gen, global_elite_limit


# ============================================================
# Prior knowledge (unchanged)
# ============================================================
prior_knowledge = [{
    'link_lengths': [0.008849317757047144, 1.465181316077158, 1.525969366165795],
    'joint_limits': [(-3.104323811217068, 3.104323811217068), (-2.2135476717406846, -2.0770894991977737), (-3.0424740198157405, 0.028968517296451335)]
},
{
    'link_lengths':  [1.5009454892914675, 0.031995269488466496, 1.4670592412200663],
    'joint_limits': [(-3.037248133500475, 3.037248133500475), (0.6853974104719596, 1.6355528728695892), (-2.8475873054248075, 2.9814270756074532)]
},
{
    'link_lengths':  [0.008849317757047144, 1.465181316077158, 1.525969366165795],
    'joint_limits': [(-3.037248133500475, 3.037248133500475), (0.6853974104719596, 1.6355528728695892), (-2.8475873054248075, 2.9814270756074532)]
},
{
    'link_lengths':  [1.3943492275174934, 0.04894091088791853, 1.556709861594588],
    'joint_limits': [(-2.7270335266737122, 2.7270335266737122), (1.0133219258739512, 2.345247247091118), (-1.9665329001925376, 2.8793604157088524)]
},
{
    'link_lengths':   [1.3943492275174934, 0.04894091088791853, 1.556709861594588],
    'joint_limits': [(-2.9316925356550825, 2.9316925356550825), (-2.564090259192487, 2.1848148392106856), (2.822451569149128, 2.854758928167925)]
},
{
    'link_lengths':  [1.2543282168145604, 0.08712226598195455, 1.6585495172034854],
    'joint_limits':[(-3.089895724724269, 3.089895724724269), (0.2700559032760741, 1.074617722653866), (-1.5161006043391705, 2.6474599586822194)]
},
{
    'link_lengths':   [1.4644359570957466, 0.04698977532204848, 1.4885742675822051],
    'joint_limits':[(-1.4706632955724315, 1.4706632955724315), (-1.5768414877131869, 1.3524303899971144), (0.21929326925655435, 2.252802982198095)]
},
{
    'link_lengths':   [1.4644359570957466, 0.04698977532204848, 1.4885742675822051],
    'joint_limits':[(-1.4307033485617333, 1.4307033485617333), (-2.234873101042512, 0.3017137290665808), (2.7125452145408033, 2.7546923166842316)]
}
]


# ============================================================
# Main GA
# ============================================================
if __name__ == "__main__":
    samples = prior_knowledge[:]
    need = sample_number - len(samples)
    if need > 0:
        samples += [generate_planar_3r_params() for _ in range(need)]

    df = pd.DataFrame(samples)
    current = df
    global_elite_limit = 0.0

    for g in range(num_generations):
        print(f"=== Generation {g} ===")
        current, global_elite_limit = next_generation(
            current, global_elite_limit, alpha
        )
        gc.collect()
        print(f"Generation {g} completed, {g+1} generated.\n")

    gc.collect()

    current, global_elite_limit = connectivity_analysis(
        current, global_elite_limit
    )
