import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


n = 50
capacity = 30


p_values = np.linspace(0, 1, 100)


E_X_values = []
for p in p_values:
    prob_X = binom.pmf(np.arange(0, capacity+1), n, p)
    E_X = np.sum(np.arange(0, capacity+1) * prob_X)
    E_X += capacity * (1 - binom.cdf(capacity, n, p))
    E_X_values.append(E_X)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(p_values, E_X_values, label=r'$E[X](p)$', color='b', linewidth=2)
plt.xlabel('Probability p')
plt.ylabel('Expected Number of Cars in Parking Lot')
plt.title('Expected Number of Cars as a Function of p')
plt.grid()
plt.legend()
plt.show()