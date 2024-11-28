import numpy as np 

def read_2bit(feature : np.ndarray[int]) -> np.ndarray[int]:
    """
    For a given 2 bit feature, read all features
    """

    feature_values = np.zeros_like(feature) - 1.
    good = (feature > 0)
    feature_values[good] = np.floor(np.log(feature[good])/np.log(2))

    return feature_values