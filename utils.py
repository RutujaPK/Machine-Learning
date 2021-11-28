import numpy as np


def softmax(x, axis):
    """
    Implements a *stabilized* softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    Do not use scipy to implement this function!
    """
    x = np.atleast_2d(x)
    numerator = np.exp(x-np.max(x, axis=axis))
    denominator = np.exp(x-np.max(x, axis=axis)).sum(axis=axis)
    
    softmax_value = numerator / denominator

    return softmax_value
    #raise NotImplementedError
