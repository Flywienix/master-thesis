import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from my_plot import set_size

plt.style.use("seaborn-v0_8-colorblind")
plt.style.use("plotting/stylesheet.mplstyle")
color_palette = sns.color_palette()

width = 432.48195

Tamb_Csb = np.load("data/Tamb_Csb.npy")
sol_Csb = np.load("data/sol_Csb.npy")
start, end = 0, Tamb_Csb.shape[0]
print(end)
# start = 1000
# end = start + 5 * 96
fig, ax = plt.subplots(
    2, 1, figsize=set_size(width, subplots=(2, 1), aspect_ratio=0.3)
)  # , sharex=True)
ax[0].plot(
    Tamb_Csb[start:end],
    label="Tamb Csb",
)
ax[1].plot(sol_Csb[start:end], label="sol Csb", color=color_palette[2])
# ax[0].set_title("Typical meteorological weather data for Csb climate type")
ax[0].set_ylabel("Ambient air\ntemperature in °C\n")
ax[1].set_ylabel("Solar radiation\nin W/m²")
ax[1].set_xlabel("Time in 15 min steps (i.e. 96 = 24 hours)")
fig.savefig("plotting/plots/weather_data_Csb.pdf", format="pdf")
