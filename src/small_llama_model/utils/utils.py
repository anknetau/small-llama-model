#pyright: strict

from utils.common import *
import sys

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

def dump_tensor_slice(tensor: SomeNPArray, name: str, start: int|None=None, end:int|None=None) -> str:
    start = start or (0,) * tensor.ndim # type: ignore
    end = end or tuple(min(dim, 10) for dim in tensor.shape)  # Default to the first 10 elements of each dimension # type: ignore
    slices = tuple(slice(s, e) for s, e in zip(start, end)) # type: ignore
    sliced_tensor = tensor[slices]
    # print(f"Shape: {tensor.shape}, Sliced Shape: {sliced_tensor.shape}")
    # print(f"Values:\n{sliced_tensor}")
    return "\n>>> {" + name + "@" + str(tensor.shape) + "/" + str(sliced_tensor).replace("\n", "") + "}\n\n"

def print_single(d: Any):
    print(d, end="")
    sys.stdout.flush()
