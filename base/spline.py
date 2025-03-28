#
# Create a clamped cubic B-spline for a given sequence of points
#

import numpy as np
from numpy.linalg import norm


Bezier = list[np.ndarray]  # always a list of four elements


def midpoint(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * (p + q)


def thirdpoint(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (1.0 / 3.0) * ((2 * p) + q)


# given the list of control points,
# compute bezier curves for a clamped cubic B-spline
def bspline(v: list[np.ndarray]) -> list[Bezier]:
    n = len(v)
    k = n - 3
    # Interval 0-1: [0,0,0], [0,0,1], [0,1,1], [1,1,1]
    q0 = v[0]  # [0,0,0]
    q1 = v[1]  # [0,0,1]
    q2 = midpoint(q1, v[2])  # [0,1,1] = 1/2 [0,1,0] + 1/2 [0,1,2]
    r = thirdpoint(v[2], v[3])  # [1,2,1] = 2/3 [0,1,2] + 1/3 [3,1,2]
    q3 = midpoint(q2, r)  # [1,1,1] = 1/2 [1,0,1] + 1/2 [1,2,1]
    result = [[q0, q1, q2, q3]]
    for i in range(1, k - 2):
        # Interval i-i+1: [i,i,i], [i,i,i+1], [i,i+1,i+1], [i+1,i+1,i+1]
        q0 = q3  # [i,i,i]
        q1 = r  # [i,i,i+1]
        # [i,i+1,i+1] = 1/2 [i,i+1,i] + 1/2 [i, i+1, i+2]
        q2 = midpoint(q1, v[i + 2])
        # [i+1,i+1,i+2] = 2/3 [i,i+1,i+2] + [i+3,i+1,i+2]
        r = thirdpoint(v[i + 2], v[i + 3])
        # [i+1,i+1,i+1] = 1/2 [i+1,i+1,i] + 1/2 [i+1,i+1,i+2]
        q3 = midpoint(q2, r)
        result.append([q0, q1, q2, q3])
    # Interval (k-2)-(k-1):
    # [k-2,k-2,k-2], [k-2,k-2,k-1], [k-2,k-1,k-1], [k-1,k-1,k-1]
    q0 = q3
    q1 = r  # [k-2,k-2,k-1]
    # [k-2,k-1,k-1] = 1/2 [k-2,k-1,k-2] + 1/2 [k-2,k-1,k]
    q2 = midpoint(q1, v[k])
    # [k-1,k-1,k] =  1/2 [k-2,k-1,k] + 1/2 [k,k-1,k]
    r = midpoint(v[k], v[k + 1])
    # [k-1,k-1,k-1] = 1/2 [k-1,k-1,k-2] + 1/2 [k-1,k-1,k]
    q3 = midpoint(q2, r)
    result.append([q0, q1, q2, q3])
    # Interval (k-1)-k: [k-1,k-1,k-1], [k-1,k-1,k], [k-1,k,k], [k,k,k]
    q0 = q3
    q1 = r
    q2 = v[n - 2]  # [k-1,k,k]
    q3 = v[n - 1]  # [k,k,k]
    result.append([q0, q1, q2, q3])
    return result


def subdivideOnce(bezier: Bezier) -> tuple[Bezier, Bezier]:
    l1 = midpoint(bezier[0], bezier[1])
    h = midpoint(bezier[1], bezier[2])
    l2 = midpoint(l1, h)
    r2 = midpoint(bezier[2], bezier[3])
    r1 = midpoint(h, r2)
    r0 = midpoint(l2, r1)
    return [bezier[0], l1, l2, r0], [r0, r1, r2, bezier[3]]


# subdivide a bezier until pieces are no longer than maxLength
def subdivide(bezier: list[np.ndarray], maxLength: float, result: list[Bezier]) -> None:
    if norm(bezier[0] - bezier[3]) < maxLength:
        result.append(bezier)
    else:
        left, right = subdivideOnce(bezier)
        subdivide(left, maxLength, result)
        subdivide(right, maxLength, result)


def subdivideBeziers(beziers: list[Bezier], maxLength: float) -> list[Bezier]:
    result = []
    for bezier in beziers:
        subdivide(bezier, maxLength, result)
    return result
