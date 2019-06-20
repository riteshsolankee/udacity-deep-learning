# import pandas as pd
# from functools import reduce
import numpy as np

import util
import data_prep

df = util.read_data("binary.csv", "../data/")

features, targets, features_test, targets_test = data_prep.prepare_data(df)

n_records, n_features = features.shape
# Use to same seed to make debugging easier
np.random.seed(42)
last_loss = None
weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

# Neural Network hyper-parameters
epochs = 10000
learn_rate = 0.1

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        output = util.sigmoid(np.dot(x, weights))

        error = y - output

        error_term = error * output * (1 - output)

        # The gradient descent step, the error times the gradient times the inputs
        del_w += error_term * x
    # Update the weights here. The learning rate times the
    # change in weights, divided by the number of records to average
    weights += learn_rate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = util.sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = util.sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))