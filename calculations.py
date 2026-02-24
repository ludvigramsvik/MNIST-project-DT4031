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