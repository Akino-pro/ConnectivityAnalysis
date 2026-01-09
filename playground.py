import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps  # modern API

def color_by_reliability(values):
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = colormaps.get_cmap('rainbow')   # replaces cm.get_cmap('rainbow')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = [cmap(norm(v)) for v in values]
    return color_list, sm

_, sm = color_by_reliability([0, 1])

# vertical gradient
y = np.linspace(0, 1, 512)
gradient = y[:, np.newaxis]

fig, ax = plt.subplots(figsize=(1.8, 6))
ax.imshow(gradient, aspect='auto', cmap=sm.cmap, norm=sm.norm)
ax.set_axis_off()

ticks = np.arange(0.0, 1.01, 0.1)

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.25, ticks=ticks)
cbar.set_label("reliability", fontsize=12)

# --- tick size ×1.2 (robust) ---
base = plt.rcParams['ytick.labelsize']
base_pts = plt.rcParamsDefault['ytick.labelsize'] if isinstance(base, str) else float(base)
cbar.ax.tick_params(labelsize=15)

plt.show()