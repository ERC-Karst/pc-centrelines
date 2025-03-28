#
# Simplify a path
#


import numpy as np
from numpy.linalg import norm


def findOutlier(
    delta: int, start: np.ndarray, vias: list[np.ndarray], target: np.ndarray
) -> int | None:
    d = target - start
    d = d * (1.0 / norm(d))
    maxI = None
    dist = 0
    for i in range(len(vias)):
        v = vias[i] - start
        pv = d * np.dot(v, d)
        dv = norm(pv - v)
        if dv > dist:
            maxI = i
            dist = dv
    if dist > delta:
        return maxI
    return None


# returns simplified path including start, excluding target
def simplify(
    delta: int, start: np.ndarray, vias: list[np.ndarray], target: np.ndarray
) -> list[np.ndarray]:
    mid = findOutlier(delta, start, vias, target)
    if mid is None:
        return [start]
    p1 = simplify(delta, start, vias[:mid], vias[mid])
    p2 = simplify(delta, vias[mid], vias[mid + 1 :], target)
    return p1 + p2


def douglasPeucker(delta: float, p1: list[np.ndarray]) -> list[np.ndarray]:
    p2 = simplify(delta, p1[0], p1[1:-1], p1[-1])
    p2.append(p1[-1])
    return p2
