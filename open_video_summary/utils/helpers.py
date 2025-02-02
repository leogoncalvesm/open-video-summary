from typing import Iterable


def custom_cosine(v1: Iterable, v2: Iterable):
    v1, v2 = list(v1), list(v2)
    return sum(v1[i] * v2[i] for i in range(len(v1)))
