import numpy as np
# variance
def sample_mean(data):
    return np.mean(data)

def total_mean(data):
    return np.mean(sample_mean(data))

def variance(data):
    return np.mean(np.square(data - total_mean(data)))

# standard deviation
def standard_deviation(data):
    return np.sqrt(variance(data))

# the most extreme outlier
def outlier(data):
    mean = total_mean(data)
    outlier = 0
    for i in data:
        if np.square(sample_mean(i)) + np.square(mean) > outlier:
            outlier = i
    return outlier