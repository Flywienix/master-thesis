import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys

# exact path to reward.py may differ
# export PYTHONPATH=~/programming/master-thesis/mt-assmuth/final_implementation/
from reward import piecewise_linear, piecewise_quadratic
from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

start, end = -2.5, 2.5
x = np.linspace(start, end, 500)

fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=0.5))
ax.plot(x, piecewise_quadratic(x), label="piecewise quadratic")
ax.plot(x, piecewise_linear(x), label="piecewise linear")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.legend()
fig.savefig("plotting/plots/reward_term_functions.pdf", format="pdf")
