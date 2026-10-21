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


"""Main tests"""


@pytest.mark.parametrize("test_input,expected", [((P, Q), 1), ((Q, P), 1)])
def test_frdist(test_input, expected):
    assert pcc.frechet_dist.frdist(
        *test_input) == pytest.approx(expected, abs=1e-4)
