import pytest
import numpy as np
import pc_centrelines as pcc

P = np.array([[1, 0],
              [2, 0],
              [3, 0],
              [4, 0],
              [5, 0]])

Q = np.array([[1, 1],
              [2, 1],
              [3, 1],
              [4, 1],
              [5, 1]])


P1 = np.array([[0, 0],
               [2, 0],
               [4, 0]])
Q1 = np.array([[3, 1],
               [2, 1],
               [1, 1]])
# Q1 is shorter than P1, as well as flipped.
# We expect trim adjust to flip Q1, trim P1, and resample both.

P1_prime_expect = np.array([[1, 0],
                            [1.5, 0],
                            [2, 0],
                            [2.5, 0],
                            [3, 0]])

Q1_prime_expect = np.array([[1, 1],
                            [1.5, 1],
                            [2, 1],
                            [2.5, 1],
                            [3, 1]])


"""Main tests"""


@pytest.mark.parametrize("test_input,expected", [((P, Q), 1), ((Q, P), 1), ((Q[::-1], P), np.sqrt(17))])
def test_frdist(test_input, expected):
    assert pcc.frechet_dist.frdist(
        *test_input) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("test_input,expected", [((P1, Q1, 0.5), (P1_prime_expect, Q1_prime_expect))])
def test_trim_adjust(test_input, expected):
    P1prime, Q1prime = pcc.utils.trim_adjust(
        *test_input)
    P1, Q1 = expected
    print(P1prime, P1)
    print(Q1prime, Q1)

    assert P1prime == pytest.approx(P1, abs=1e-4)
    assert Q1prime == pytest.approx(Q1, abs=1e-4)
