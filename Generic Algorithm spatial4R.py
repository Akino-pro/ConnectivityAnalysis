import numpy as np
import os
#from spatial_4r_FTW_morphological_estimation_ssm import ssm_estation


# =========================================================
# GA HYPERPARAMETERS
# =========================================================
POP_SIZE = 95
NUM_GENERATIONS = 1
ALPHA = 0.05                   # elitism ratio
TOURNAMENT_K = 3              # tournament selection size
MUTATION_STD = {
    'alpha': np.deg2rad(3),   # ~3 deg
    'a': 0.03,
    'd': 0.05,
    'theta': np.deg2rad(4)    # ~4 deg
}
GRID_SAMPLE_NUMBER = 72
SAVE_FILE = "generation_population.npy"


# =========================================================
# INDIVIDUAL STRUCTURE
# =========================================================
def generate_individual():
    """Generate a valid spatial 4R parameter set."""
    ind = {}

    ind['alpha'] = np.array([85, -53, -89, 68]) * np.pi / 180.0
    ind['a'] = np.array([0.50, 0.48, 0.76, 0.95])
    ind['d'] = np.array([-0.29, 0.00, 0.05, 1.00])

    theta = np.zeros((4, 2))
    for i in range(4):
        lo = np.random.uniform(-np.pi, np.pi)
        hi = np.random.uniform(-np.pi, np.pi)
        theta[i, 0] = min(lo, hi)
        theta[i, 1] = max(lo, hi)
    ind['theta'] = theta

    ind['fitness'] = None
    return ind


def copy_individual(ind):
    return {
        'alpha': np.array(ind['alpha']),
        'a': np.array(ind['a']),
        'd': np.array(ind['d']),
        'theta': np.array(ind['theta']),
        'fitness': ind['fitness']
    }


# =========================================================
# FILE STORAGE
# =========================================================
def save_population(pop):
    """Save entire population to disk."""
    serializable = []
    for ind in pop:
        serializable.append({
            'alpha': ind['alpha'],
            'a': ind['a'],
            'd': ind['d'],
            'theta': ind['theta'],
            'fitness': ind['fitness']
        })
    np.save(SAVE_FILE, serializable, allow_pickle=True)
    print(f"✔ Saved population to {SAVE_FILE}")


def load_population():
    """Load population from disk if available."""
    if not os.path.exists(SAVE_FILE):
        print("⚠ No saved population found → starting from scratch")
        return None

    print(f"✔ Loading population from {SAVE_FILE}")
    data = np.load(SAVE_FILE, allow_pickle=True)

    pop = []
    for item in data:
        pop.append({
            'alpha': np.array(item['alpha']),
            'a': np.array(item['a']),
            'd': np.array(item['d']),
            'theta': np.array(item['theta']),
            'fitness': item['fitness']
        })
    return pop


# =========================================================
# FITNESS
# =========================================================
def evaluate(ind):
    ind['fitness'] = ssm_estimation(
        GRID_SAMPLE_NUMBER,
        ind['d'],
        ind['alpha'],
        ind['a'],
        ind['theta']
    )
    return ind['fitness']


def evaluate_population(pop):
    for ind in pop:
        evaluate(ind)


# =========================================================
# SELECTION
# =========================================================
def tournament_select(pop, k=TOURNAMENT_K):
    cand = np.random.choice(pop, k)
    return max(cand, key=lambda x: x['fitness'])


# =========================================================
# CROSSOVER
# =========================================================
def crossover(pa, pb):
    ca = copy_individual(pa)
    cb = copy_individual(pb)

    for i in range(4):
        if np.random.rand() < 0.5:
            ca['alpha'][i], cb['alpha'][i] = pb['alpha'][i], pa['alpha'][i]
        if np.random.rand() < 0.5:
            ca['a'][i], cb['a'][i] = pb['a'][i], pa['a'][i]
        if np.random.rand() < 0.5:
            ca['d'][i], cb['d'][i] = pb['d'][i], pa['d'][i]

    for i in range(4):
        for j in range(2):
            if np.random.rand() < 0.5:
                ca['theta'][i, j], cb['theta'][i, j] = pb['theta'][i, j], pa['theta'][i, j]

        lo, hi = ca['theta'][i]
        ca['theta'][i] = np.array(sorted([lo, hi]))

        lo, hi = cb['theta'][i]
        cb['theta'][i] = np.array(sorted([lo, hi]))

    ca['fitness'] = None
    cb['fitness'] = None
    return ca, cb


# =========================================================
# MUTATION
# =========================================================
def mutate(ind):
    ind = copy_individual(ind)

    ind['alpha'] += np.random.randn(4) * MUTATION_STD['alpha']

    ind['a'] += np.random.randn(4) * MUTATION_STD['a']
    ind['a'] = np.clip(ind['a'], 0, 1)

    ind['d'] += np.random.randn(4) * MUTATION_STD['d']
    ind['d'] = np.clip(ind['d'], -1, 1)

    ind['theta'] += np.random.randn(4, 2) * MUTATION_STD['theta']
    for i in range(4):
        lo, hi = ind['theta'][i]
        ind['theta'][i] = np.array(sorted([lo, hi]))

    ind['fitness'] = None
    return ind


# =========================================================
# MAIN GA STEPS
# =========================================================
def create_population():
    return [generate_individual() for _ in range(POP_SIZE)]


def elitism(pop):
    keep = int(POP_SIZE * ALPHA)
    pop_sorted = sorted(pop, key=lambda x: x['fitness'], reverse=True)
    return pop_sorted[:keep]


def next_generation(pop):
    evaluate_population(pop)

    elites = elitism(pop)
    new_pop = elites[:]

    while len(new_pop) < POP_SIZE:
        pa = tournament_select(pop)
        pb = tournament_select(pop)

        ca, cb = crossover(pa, pb)

        if np.random.rand() < 0.15:
            ca = mutate(ca)
        if np.random.rand() < 0.15:
            cb = mutate(cb)

        new_pop.append(ca)
        if len(new_pop) < POP_SIZE:
            new_pop.append(cb)

    return new_pop


# =========================================================
# PRIOR KNOWLEDGE
# =========================================================
prior_knowledge = [
    {
        'alpha': [85*np.pi/180, -53*np.pi/180, -89*np.pi/180, 68*np.pi/180],
        'a': [0.5, 0.48, 0.76, 0.95],
        'd': [-0.29, 0, 0.05, 1],
        'theta': [
            [-0.86158053, 0.86158053],
            [-3.85579283, 1.65596551],
            [-3.01203879, -0.13893174],
            [0.67830055, 2.19514936]
        ]
    }
]


def inject_prior(pop):
    for i, prior in enumerate(prior_knowledge):
        if i >= len(pop):
            break
        pop[i]['alpha'] = np.array(prior['alpha'])
        pop[i]['a'] = np.array(prior['a'])
        pop[i]['d'] = np.array(prior['d'])
        pop[i]['theta'] = np.array(prior['theta'])
        pop[i]['fitness'] = None


# =========================================================
# RUN GA WITH PERSISTENCE
# =========================================================
def run_ga():
    # Try loading previous population
    pop = load_population()
    if pop is None:
        print("→ Creating new population")
        pop = create_population()
        inject_prior(pop)

    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generation {gen} ===")
        pop = next_generation(pop)

        # Save population
        save_population(pop)

        best = max(pop, key=lambda x: x['fitness'])
        print(f"Best fitness = {best['fitness']:.6f}")

    print("\nFINAL CHAMPION:")
    print(best)
    return pop, best


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    final_pop, champion = run_ga()
