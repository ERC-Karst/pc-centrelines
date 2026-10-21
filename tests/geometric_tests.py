import pytest 
import numpy as np
import pc_centrelines as pcc


def square(s, reverse=False):
    if reverse == True:
        return np.array(([0, 0],[0, s], [s, s], [s, 0]))
    else: 
        return np.array(([0, 0],[s, 0], [s, s], [0, s]))


a = square(1)
b = square(2)
c = (square(3, reverse=True))

def ellipse(a, b, theta, npoints = 2000):
    thetai = np.linspace(-np.pi, np.pi, npoints)
    return np.array([a*np.cos(thetai +theta), b*np.sin(thetai+theta)]).T

d = ellipse(1, 1, 0)
e = ellipse(2, 1, 0)
f = ellipse(1, 2, np.pi/2)

"""Main tests"""

@pytest.mark.parametrize("test_input,expected", [(a, 1), (b, 4), (c,9), (d, np.pi), (e, 2*np.pi), (f, 2*np.pi)])
def test_area(test_input, expected): 
    assert pcc.geometry.compute_ngon_area(test_input) == pytest.approx(expected, abs= 1e-4)

@pytest.mark.parametrize("test_input,expected", [(a, 4), (b, 8), (c,12), (d, 2*np.pi)])
def test_perimeter(test_input, expected): 
    assert pcc.geometry.compute_ngon_perimeter(test_input) == pytest.approx(expected, abs = 1e-4)