#pyright: strict

from utils.common import *

SomeNPArray: TypeAlias = np.ndarray[Any, Any]

def silu(x: SomeNPArray):
    return x / (1.0 + np.exp(-x))

def apply_temp(z: SomeNPArray, temperature: float):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    return z / temperature

def softmax(x: SomeNPArray, axis:int|None=-1, keepdims:bool=True):
    maxdiff = x - np.max(x, axis=axis, keepdims=keepdims)
    exp_x = np.exp(maxdiff)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=keepdims)

def softmax_last(x: SomeNPArray):
    return softmax(x)

def softmax_all(x: SomeNPArray):
    return softmax(x, axis=None, keepdims=True)
