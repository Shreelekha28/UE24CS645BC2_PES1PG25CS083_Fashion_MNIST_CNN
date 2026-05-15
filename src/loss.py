import numpy as np

def cross_entropy_loss(predictions, label):

    return -np.log(predictions[label])