from utils import (
    load_model,
    load_weather_data,
    load_price_data,
    load_full_weather_data,
    load_full_price_data,
)
from MPC import setup_MPC, solve_MPC

import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# ------- load data -------
# climate_zone = "Csb"
# solar, Tamb = load_weather_data(climate_zone)
solar, Tamb = load_full_weather_data()

# ------- load model -------
model_number = 18
model = load_model(model_number)

# ------- load price data -------
# price = load_price_data()
price = load_full_price_data()

# ------- define environment -------
environment_number = 18
environment = load_model(environment_number)

# ------- MPC parameters -------
N = 96  # prediction horizon (normally something like 1 day = 96 time steps when assuming 15 min steps).
X0 = [21, 21, 21, 21, 21]
X_env = X0  # environment uses the same initial condition

# ------- Storage for results -------
all_y = []
all_y.append(X0[0])
all_u = []

# ------- MPC -------
n_days = 168  # 3 (originally) # for how many days do we want to run the MPC?
for k in range(n_days * 96):
    m, y, x, u = setup_MPC(N, model, price, solar, Tamb, X0)
    sol = solve_MPC(m)

    # Apply the environment to obtain the new states
    X0 = environment["A"] @ X0 + environment["B"] @ sol.value(u)[:, 0]

    all_y.append(X0[0])
    all_u.append(sol.value(u[0]))

    # shift the index by 1 (as one time step has passed)
    price = price[1:]
    solar = solar[1:]
    Tamb = Tamb[1:]

all_y = np.array(all_y)
all_u = np.array(all_u)
np.save("mpc_temp", all_y)
np.save("mpc_action", all_u)

plot_start = 8 * 96
plot_duration = 3 * 96
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# price = load_price_data()  # load unshifted price signal for plotting
price = load_full_price_data()
ax[0].plot(all_y[plot_start : plot_start + plot_duration])
ax[0].set_ylabel("Indoor air temperature in °C")
ax[1].plot(price[plot_start : plot_start + plot_duration])
ax[1].set_ylabel("Price data")
ax[1].set_xlabel("Time in 15 min steps")
ax2 = ax[1].twinx()
ax2.plot(all_u[plot_start : plot_start + plot_duration], color="red")
plt.show()

1 + 1
