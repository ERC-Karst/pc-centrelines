import pytest 
import numpy as np
import pc_centrelines as pcc

from os import chdir, path, curdir
thepath = path.abspath("../base")
print(curdir)
chdir(thepath)

print(curdir)

from pcc.geometry import compute_ngon_area

@pytest.fixture
def square(s, reverse=False):
    if reverse == True:
        return np.array([0, 0],[0, s], [s, s], [s, 0])
    else: 
        return np.array([0, 0],[s, 0], [s, s], [0, s])

    
def test_area(): 
    assert compute_ngon_area(square(1)) == pytest.approx(1)
    assert compute_ngon_area(square(2)) == pytest.approx(4)
    assert compute_ngon_area(square(3, reverse=True)) == pytest.approx(9)