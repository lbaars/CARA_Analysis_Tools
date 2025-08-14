import pytest
import numpy as np
import numpy.testing as npt
import cara_analysis_tools.pc.utils as pcu

def test_product3x3_not_np_array():
    badArray = "asdf"
    goodArray = np.full(shape = (1,9), fill_value = np.nan)
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.product3x3(badArray, goodArray)
    with pytest.raises(ValueError):
        pcu.product3x3(goodArray, badArray)
    
    # Test success
    pcu.product3x3(goodArray, goodArray)

def test_product3x3_not_2D_array():
    badArray = np.array([1, 2])
    goodArray = np.full(shape = (1,9), fill_value = np.nan)
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.product3x3(badArray,goodArray)
    with pytest.raises(ValueError):
        pcu.product3x3(goodArray,badArray)
    
    # Test success
    pcu.product3x3(goodArray, goodArray)

def test_product3x3_bad_number_columns():
    badArray = np.array([[1, 2, 3]])
    goodArray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.product3x3(badArray,goodArray)
    with pytest.raises(ValueError):
        pcu.product3x3(goodArray,badArray)
    
    # Test success
    pcu.product3x3(goodArray, goodArray)

def test_product3x3_mismatch_rows():
    array1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    array2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.product3x3(array1,array2)
    with pytest.raises(ValueError):
        pcu.product3x3(array2,array1)
    
    # Test success
    pcu.product3x3(array1, array1)
    pcu.product3x3(array2, array2)

def test_product3x3_static_test():
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    b = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10]])
    expVal = np.array([[36, 42, 48, 81, 96, 111, 126, 150, 174]])
    
    actVal = pcu.product3x3(a, b)
    npt.assert_equal(actVal, expVal)

def test_product3x3_random_calcs():
    np.random.seed(1)
    numTests = 1000
    a = np.random.uniform(low = -1000.0, high = 1000.0, size=(numTests,9))
    b = np.random.uniform(low = -1000.0, high = 1000.0, size=(numTests,9))
    
    p = pcu.product3x3(a, b)
    for i in range(numTests):
        m_act = p[i,:].reshape((3, 3))
        aTest = a[i,:].reshape((3, 3))
        bTest = b[i,:].reshape((3, 3))
        m_exp = aTest @ bTest
        npt.assert_allclose(m_act, m_exp)
        
        