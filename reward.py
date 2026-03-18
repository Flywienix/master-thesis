import numpy as np


def heatpump_reward_function(
    temp_weight: float,
    price_weight: float,
    target_temperature: float,
    temp_term_function: callable,
) -> callable:
    return lambda current_temperature, current_price, current_action: (
        -temp_weight * temp_term_function(target_temperature - current_temperature),
        -price_weight * current_price * current_action,
    )


def piecewise_linear(
    x,
    left: float = -1.0,
    right: float = 1.0,
    slope: float = 1.0,
    transition: float = 0.01,
):

    conditions = [
        x <= left - transition,
        (left - transition < x) & (x < left),
        (left <= x) & (x <= right),
        (right < x) & (x < right + transition),
        x >= right + transition,
    ]

    functions = [
        lambda x: -(-slope / 2) * transition - slope * (x - (left - transition)),
        lambda x: -(-slope / (2 * transition)) * (x - left) ** 2,
        0,
        lambda x: (slope / (2 * transition)) * (x - right) ** 2,
        lambda x: (slope / 2) * transition + slope * (x - (right + transition)),
    ]

    return np.piecewise(x, conditions, functions)


def piecewise_quadratic(
    x,
    left: float = -1.0,
    right: float = 1.0,
):

    conditions = [
        x <= left,
        (left <= x) & (x <= right),
        right < x,
    ]

    functions = [
        lambda x: (x - left) ** 2,
        0,
        lambda x: (x - right) ** 2,
    ]

    return np.piecewise(x, conditions, functions)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-2.5, 2.5, 500)
    # x = np.linspace(0.9, 1.1, 500)
    y_l = piecewise_linear(x)
    plt.plot(x, y_l)
    y_q = 10 * piecewise_quadratic(x)
    plt.plot(x, y_q)
    plt.show()  # if not working pip install PyQt6
