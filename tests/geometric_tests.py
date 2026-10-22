import pytest
import numpy as np
import pc_centrelines as pcc


def square(s, reverse=False):
    if reverse == True:
        return np.array(([0, 0], [0, s], [s, s], [s, 0]))
    else:
        return np.array(([0, 0], [s, 0], [s, s], [0, s]))

    


a = square(1)
b = square(2)
c = (square(3, reverse=True))


def ellipse(a, b, theta, npoints=2000):
    thetai = np.linspace(-np.pi, np.pi, npoints)
    return np.array([a*np.cos(thetai + theta), b*np.sin(thetai+theta)]).T


d = ellipse(1, 1, 0)
e = ellipse(2, 1, 0) + np.array([-1, 1])
f = ellipse(1, 2, np.pi/2) + np.array([2, 7])

# make the b square concave by adding reentrants.
b1 = np.vstack((b, [1, 1]))

d_cut = pcc.geometry.return_below_fill_height(d, 0.5)[0]

"""Main tests"""


@pytest.mark.parametrize("test_input,expected", [(a, 1), (b, 4), (c, 9), (d, np.pi), (e, 2*np.pi), (f, 2*np.pi), (d_cut, np.pi/2)])
def test_area(test_input, expected):
    assert pcc.geometry.compute_ngon_area(
        test_input) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("test_input,expected", [(a, 4), (b, 8), (c, 12), (d, 2*np.pi), (d_cut, np.pi + 2)])
def test_perimeter(test_input, expected):
    assert pcc.geometry.compute_ngon_perimeter(
        test_input) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("test_input,expected", [((1, 4), 1), ((4, 8), 2), ((9, 12), 3), ((np.pi, 2*np.pi), 2)])
def test_Dh(test_input, expected):
    assert pcc.geometry.compute_ngon_dh(
        *test_input) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("test_input,expected", [(b1, b)])
def test_convexhull(test_input, expected):
    assert pcc.geometry.compute_convex_hull(
        test_input) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("test_input,expected", [(a, (1, 1)), (b, (4, 1)), (c, (9, 1)), (d, (np.pi, 1)), (e, (2*np.pi, 2)), (f, (2*np.pi, 2))])
def test_ellipse(test_input, expected):
    _, ellipse_area, _, aspect_ratio, _, _, _ = pcc.geometry.fit_ellipse(
        test_input, npoints=4000)
    assert (ellipse_area, aspect_ratio) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("test_input,expected", [(a, [0.5, 0.5]), (b, (1,1)), (c, (1.5, 1.5)), (d, (0,0)), (e, (-1,1)), (f, (2,7))])
def test_centre_of_gravity(test_input, expected):
    assert pcc.centre_of_gravity(test_input) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("test_input,expected", [(a, np.sqrt(2)/2), (b, np.sqrt(2)), (c, 3*np.sqrt(2)/2), (d, 1), (e, 2), (f, 2)])
def test_rmax(test_input, expected):
    c = pcc.centre_of_gravity(test_input)
    rmax, _ = pcc.compute_rmax(test_input, c)
    assert rmax == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("test_input,expected", [(a, np.sqrt(2)), (b, 2*np.sqrt(2)), (c, 3*np.sqrt(2)), (d, 2), (e, 4), (f, 4)])
def test_maxpairwise_distance(test_input, expected):
    assert pcc.compute_max_pairwise_distance(
        test_input) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("test_input,expected", [(a, 1), (b, 1), (c, 1), (d, 1), (e, 1), (f, 1), (d_cut, 1)])
def test_solidity(test_input, expected):
    assert pcc.compute_polygon_solidity(
        test_input) == pytest.approx(expected, abs=1e-3)
