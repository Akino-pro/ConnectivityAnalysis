#!/usr/bin/env python3
"""
EE576 HW‑6 – Plot connection durations (axes reversed)
Creates two figures:
  • Figure 1 – SYN‑cookies OFF
  • Figure 2 – SYN‑cookies ON
x‑axis  = connection duration (s)
y‑axis  = connection start order (≈ time)
Vertical dashed lines at 30 s and 150 s show the attack window.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# File names – change if yours differ
nocookie_file = Path("nospoof.txt")
cookie_file   = Path("cookie.txt")

# ------------------------------------------------------------------
def load_durations(path: Path) -> pd.Series:
    """Return a Series of floats containing connection durations."""
    return (pd.read_csv(path, sep=r"\s+", usecols=[0], header=None)
              .astype(float)
              .squeeze())

dur_no  = load_durations(nocookie_file)
dur_yes = load_durations(cookie_file)

# Use the row index (0, 1, 2, …) as connection‑order proxy
dur_no.index  = range(len(dur_no))
dur_yes.index = range(len(dur_yes))

ATTACK_START = 30    # seconds after tcpdump began
ATTACK_END   = 150   # attack stops

# ------------------------------------------------------------------
def make_plot(series, title, fname):
    """Create one figure with x = duration, y = connection order."""
    plt.figure(figsize=(8, 5))
    plt.plot(series.values, series.index, linewidth=1.2)
    # Vertical lines marking attack window (x = duration axis)
    plt.axvline(ATTACK_START, linestyle="--", color="red", linewidth=0.8)
    plt.axvline(ATTACK_END,   linestyle="--", color="red", linewidth=0.8)
    plt.text(ATTACK_START, max(series.index)*0.98, "attack start",
             rotation=90, va="top", ha="right", color="red", fontsize=8)
    plt.text(ATTACK_END,   max(series.index)*0.98, "attack end",
             rotation=90, va="top", ha="left",  color="red", fontsize=8)

    plt.title(title)
    plt.xlabel("Connection duration (s)")
    plt.ylabel("Connection start order (≈ time)")
    plt.xlim(left=0)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)     # PNG for your report
    plt.show()

# ------------------------------------------------------------------
make_plot(dur_no,  "Connection durations – SPOOF OFF", "durations_off_xyrev.png")
make_plot(dur_yes, "Connection durations – SYN‑cookies ON",  "durations_on_xyrev.png")
