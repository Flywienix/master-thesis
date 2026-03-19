import numpy as np
import scipy.io


def load_model(model_number):
    # let's load an exemplary model. Note that this assumes that you root directory is mt-assmuth
    mat = scipy.io.loadmat("../data/building models/models_multi_matrices.mat")

    models = mat["models_multi_matrices"]["models"][0, 0]
    # remove empty arrays
    models = models[0]

    model = models[model_number]
    # remove some empty arrays again...
    model = model[0][0]
    # extract the matrices to construct the state space system
    A = model[0]
    B = model[1]
    C = model[2]
    # create the matrix D full of zeros
    D = np.zeros([1, 5])

    model = {"A": A, "B": B, "C": C, "D": D}
    return model


def load_weather_data(climate_zone):
    mat = scipy.io.loadmat("../data/weather/TMYs.mat")
    GER_weather = mat["TMYs"][f"{climate_zone}"]
    weather = GER_weather[0][0][0][0][0][0][0]
    sol = weather[0]  # solar radiation
    Tamb = weather[1]  # ambient air temp.
    return sol, Tamb


def load_price_data():
    mat = scipy.io.loadmat("../data/price data/prices_validation.mat")
    prices = mat["prices"]
    price = prices[0][0][0][0][0][0]
    return price


## not original ##
def load_full_weather_data():
    sol = np.load("../final_implementation/data/sol_Csb.npy")
    Tamb = np.load("../final_implementation/data/Tamb_Csb.npy")
    return sol, Tamb


def load_full_price_data():
    price = np.load("../final_implementation/data/price_data.npy")
    return price
