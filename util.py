import pandas as pd
import numpy as np


def read_data(file_name, path="./data/"):
    return pd.read_csv(str(path + file_name), header=0)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))
