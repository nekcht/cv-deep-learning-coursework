from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    affine_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(drelu, affine_cache)
    return dx, dw, db

