import pytest
import numpy as np
from scipy.linalg import issymmetric
import cara_analysis_tools.utils.aug_math as am

def test_cov_make_symmetric_not_np_array():
    C = "asdf"
    
    with pytest.raises(ValueError):
        am.cov_make_symmetric(C)

def test_cov_make_symmetric_not_2D_np_array():
    C = np.array([0, 1])
    
    with pytest.raises(ValueError):
        am.cov_make_symmetric(C)

def test_cov_make_symmetric_not_square_matrix():
    C = np.array([[0, 1, 2], [3, 4, 5]])
    
    with pytest.raises(ValueError):
        am.cov_make_symmetric(C)

def test_cov_make_symmetric_not_at_least_2x2_matrix():
    C = np.array([[0]])
    
    with pytest.raises(ValueError):
        am.cov_make_symmetric(C)

def test_cov_make_symmetric_already_symmetric():
    C = np.array([[1.0, 2.0], [2.0, 3.0]])
    
    Csym = am.cov_make_symmetric(C)
    for i in range(2):
        for j in range(2):
            assert Csym[i,j] == C[i,j]
    
    assert issymmetric(Csym)

def test_cov_make_symmetric_make_symmetric():
    C = np.array([[1.0, 2.0], [2.1, 3.0]])
    CsymExp = np.array([[1.0, 2.05], [2.05, 3.0]])
    
    Csym = am.cov_make_symmetric(C)
    for i in range(2):
        for j in range(2):
            assert Csym[i,j] == CsymExp[i,j]
    
    assert issymmetric(Csym)

