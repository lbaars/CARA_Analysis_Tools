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
        
def test_check_and_resize_nx3_not_np_array():
    badArray = "asdf"
    goodArray = np.full(shape = (1,3), fill_value = np.nan)
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.check_and_resize_nx3(badArray)
    
    # Test success
    pcu.check_and_resize_nx3(goodArray)

def test_check_and_resize_nx3_wrong_size_array():
    badArray1 = np.array([1, 2])  # 1D array should have size 3
    badArray2 = np.array([1, 2, 3, 4]) # 1D array should have size 3
    # 3D array not allowed
    badArray3 = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 0],[1, 2]]])
    badArray4 = np.array([[1, 2]]) # 2D array should have 3 columns
    
    # Test failures
    with pytest.raises(ValueError):
        pcu.check_and_resize_nx3(badArray1)
    with pytest.raises(ValueError):
        pcu.check_and_resize_nx3(badArray2)
    with pytest.raises(ValueError):
        pcu.check_and_resize_nx3(badArray3)
    with pytest.raises(ValueError):
        pcu.check_and_resize_nx3(badArray4)

def test_check_and_resize_nx3_1d_to_2d():
    array_1d = np.array([1, 2, 3])
    
    # Test resize of 1D to 2D
    array_2d = pcu.check_and_resize_nx3(array_1d)
    a_size = np.shape(array_2d)
    npt.assert_equal(len(a_size),2)
    npt.assert_equal(a_size,np.array([1, 3]))
    npt.assert_equal(array_2d,np.array([[1, 2, 3]]))

def test_check_and_resize_nx3_2d():
    array_2d_in = np.array([[1, 2, 3]])
    
    # Test 2D doesn't change
    array_2d_out = pcu.check_and_resize_nx3(array_2d_in)
    a_size = np.shape(array_2d_out)
    npt.assert_equal(len(a_size),2)
    npt.assert_equal(a_size,np.array([1, 3]))
    npt.assert_equal(array_2d_out,array_2d_in)

def test_check_and_resize_posvel_not_np_array():
    badArray = "asdf"
    goodArray = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_posvel(badArray, goodArray)
    with pytest.raises(ValueError):
        pcu.check_and_resize_posvel(goodArray, badArray)

def test_check_and_resize_posvel_cannot_resize_v():
    a_1row = np.array([[1, 2, 3]])
    a_2rows = np.array([[2, 3, 4], [5, 6, 7]])
    a_3rows = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
    
    # Test failuers
    with pytest.raises(ValueError):
        pcu.check_and_resize_posvel(a_1row, a_2rows)
    with pytest.raises(ValueError):
        pcu.check_and_resize_posvel(a_3rows, a_2rows)
    
    # Test successes
    pcu.check_and_resize_posvel(a_1row, a_1row)
    pcu.check_and_resize_posvel(a_2rows, a_1row)
    pcu.check_and_resize_posvel(a_2rows, a_2rows)
    pcu.check_and_resize_posvel(a_3rows, a_1row)
    pcu.check_and_resize_posvel(a_3rows, a_3rows)

def test_check_and_resize_posvel_no_resize():
    r_1row = np.array([[1, 2, 3]])
    r_2rows = np.array([[1, 2, 3], [4, 5, 6]])
    v_1row = np.array([[2, 3, 4]])
    v_2rows = np.array([[2, 3, 4], [5, 6, 7]])
    
    (numRows, r_out, v_out) = pcu.check_and_resize_posvel(r_1row, v_1row)
    npt.assert_equal(numRows, 1)
    npt.assert_equal(r_out, r_1row)
    npt.assert_equal(v_out, v_1row)
    
    (numRows, r_out, v_out) = pcu.check_and_resize_posvel(r_2rows, v_2rows)
    npt.assert_equal(numRows, 2)
    npt.assert_equal(r_out, r_2rows)
    npt.assert_equal(v_out, v_2rows)

def test_check_and_resize_posvel_resize():
    r_2rows = np.array([[1, 2, 3], [4, 5, 6]])
    r_3rows = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v_1row = np.array([[2, 3, 4]])
    v_2rows = np.array([[2, 3, 4], [2, 3, 4]])
    v_3rows = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4]])
    
    # Convert 1 velocity row into 2 velocity rows
    (numRows, r_out, v_out) = pcu.check_and_resize_posvel(r_2rows, v_1row)
    npt.assert_equal(numRows, 2)
    npt.assert_equal(r_out, r_2rows)
    npt.assert_equal(v_out, v_2rows)
    
    # Convert 1 velocity row into 3 velocity rows
    (numRows, r_out, v_out) = pcu.check_and_resize_posvel(r_3rows, v_1row)
    npt.assert_equal(numRows, 3)
    npt.assert_equal(r_out, r_3rows)
    npt.assert_equal(v_out, v_3rows)

def test_check_and_resize_cov_not_np_array():
    badArray = "asdf"
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)

def test_check_and_resize_cov_1d_array_wrong_size():
    badArray = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)

def test_check_and_resize_cov_2d_array_wrong_size():
    badArray = np.array([[1, 2], [2, 3]])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)

def test_check_and_resize_cov_2d_array_and_n_size_mismatch():
    badArray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 0]])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)

def test_check_and_resize_cov_3d_array_and_n_size_mismatch():
    badArray = np.array([[[1, 2, 3],
                          [2, 3, 4],
                          [3, 4, 5]],
                         [[2, 3, 4],
                          [3, 4, 5],
                          [5, 6, 7]]])
    badArray2 = np.array([[[1, 2],
                           [2, 3]],
                          [[3, 4],
                           [4, 5]]])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(2, badArray2)

def test_check_and_resize_cov_4d_array():
    badArray = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]])
    
    with pytest.raises(ValueError):
        pcu.check_and_resize_cov(1, badArray)

def test_check_and_resize_cov_reshape_1d_array():
    inArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    expArray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    
    actArray = pcu.check_and_resize_cov(1, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_resize_cov_reshape_3x3():
    inArray = np.array([[1, 2, 3],
                        [2, 3, 4],
                        [3, 4, 5]])
    expArray = np.array([[1, 2, 3, 2, 3, 4, 3, 4, 5]])
    
    actArray = pcu.check_and_resize_cov(1, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_resize_cov_reshape_6x6():
    inArray = np.array([[1, 2, 3, 4, 5, 6],
                        [2, 3, 4, 5, 6, 7],
                        [3, 4, 5, 6, 7, 8],
                        [4, 5, 6, 7, 8, 9],
                        [5, 6, 7, 8, 9, 0],
                        [6, 7, 8, 9, 0, 1]])
    expArray = np.array([[1, 2, 3, 2, 3, 4, 3, 4, 5]])
    
    actArray = pcu.check_and_resize_cov(1, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_resize_cov_resize_1x9_to_nx9():
    inArray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    n = 3
    expArray = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    
    actArray = pcu.check_and_resize_cov(n, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_resize_cov_resize_3x3_to_nx9():
    inArray = np.array([[1, 2, 3],
                        [2, 3, 4],
                        [3, 4, 5]])
    n = 3
    expArray = np.array([[1, 2, 3, 2, 3, 4, 3, 4, 5],
                         [1, 2, 3, 2, 3, 4, 3, 4, 5],
                         [1, 2, 3, 2, 3, 4, 3, 4, 5]])
    
    actArray = pcu.check_and_resize_cov(n, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_resize_cov_resize_6x6_to_nx9():
    inArray = np.array([[1, 2, 3, 4, 5, 6],
                        [2, 3, 4, 5, 6, 7],
                        [3, 4, 5, 6, 7, 8],
                        [4, 5, 6, 7, 8, 9],
                        [5, 6, 7, 8, 9, 0],
                        [6, 7, 8, 9, 0, 1]])
    n = 3
    expArray = np.array([[1, 2, 3, 2, 3, 4, 3, 4, 5],
                         [1, 2, 3, 2, 3, 4, 3, 4, 5],
                         [1, 2, 3, 2, 3, 4, 3, 4, 5]])
    
    actArray = pcu.check_and_resize_cov(n, inArray)
    npt.assert_equal(actArray, expArray)

def test_check_and_reszie_cov_resize_3d_to_nx9():
    inArray = np.array([[[1, 2, 3, 4, 5, 6],
                         [2, 3, 4, 5, 6, 7],
                         [3, 4, 5, 6, 7, 8],
                         [4, 5, 6, 7, 8, 9],
                         [5, 6, 7, 8, 9, 0],
                         [6, 7, 8, 9, 0, 1]],
                        [[2, 3, 4, 5, 6, 7],
                         [3, 4, 5, 6, 7, 8],
                         [4, 5, 6, 7, 8, 9],
                         [5, 6, 7, 8, 9, 0],
                         [6, 7, 8, 9, 0, 1],
                         [7, 9, 9, 0, 1, 2]]])
    n = 2
    expArray = np.array([[1, 2, 3, 2, 3, 4, 3, 4, 5],
                         [2, 3, 4, 3, 4, 5, 4, 5, 6]])
    
    actArray = pcu.check_and_resize_cov(n, inArray)
    npt.assert_equal(actArray, expArray)

