from numpy import ndarray


def custom_cosine(v1: ndarray, v2: ndarray) -> float:
    return sum(v1[i] * v2[i] for i in range(len(v1)))
