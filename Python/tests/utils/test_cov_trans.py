import pytest
import numpy as np
import numpy.testing as npt
import math
import inspect
from scipy.linalg import issymmetric
import cara_analysis_tools.utils.cov_trans as ct

@pytest.fixture
def common_test_data():
    data = {
        "relTol": 1e-12,
        "r": np.array([378.39559, 4305.721887, 5752.767554]),
        "v": np.array([2.360800244, 5.580331936, -4.322349039]),
        "ric": np.array([[ 1.75574609146441e-10, -1.76391822498568e-10,  1.26701582864723e-11, -8.50798921278566e-12,  2.80465474421549e-11,  1.78603570386593e-12,  5.90834321905617e-08, 0, -1.90574918760627e-07],
                         [-1.76391822498568e-10,  2.19847186431344e-10, -2.11714974937397e-11,  4.88900720193951e-12, -2.07150417963915e-11,  2.63481195571044e-12, -8.18741459439353e-08, 0,  2.89052907957242e-07],
                         [ 1.26701582864723e-11, -2.11714974937397e-11,  1.03862044222145e-11, -8.87792456013774e-12,  3.61060060754719e-12, -8.98899082284049e-15,  1.33873845292676e-08, 0, -5.14960908209782e-08],
                         [-8.50798921278566e-12,  4.88900720193951e-12, -8.87792456013774e-12,  1.81815864084123e-11, -1.11847510974009e-11, -2.74005163036409e-12, -1.47578050245331e-08, 0,  3.62925053394288e-08],
                         [ 2.80465474421549e-11, -2.07150417963915e-11,  3.61060060754719e-12, -1.11847510974009e-11,  3.46915046391962e-11,  1.05740377170100e-11,  1.07019379593921e-08, 0, -5.41768749409939e-08],
                         [ 1.78603570386593e-12,  2.63481195571044e-12, -8.98899082284049e-15, -2.74005163036409e-12,  1.05740377170100e-11,  4.14448895239156e-12, -2.98895146350504e-09, 0, -2.96953343396525e-09],
                         [ 5.90834321905617e-08, -8.18741459439353e-08,  1.33873845292676e-08, -1.47578050245331e-08,  1.07019379593921e-08, -2.98895146350504e-09,  0.00036231          , 0, -0.00014813          ],
                         [ 0                   ,  0                   ,  0                   ,  0                   ,  0                   ,  0                   ,  0                   , 0,  0                   ],
                         [-1.90574918760627e-07,  2.89052907957242e-07, -5.14960908209782e-08,  3.62925053394288e-08, -5.41768749409939e-08, -2.96953343396525e-09, -0.00014813          , 0,  0.0011749           ]]),
        "eci": np.array([[   3.748e-11,   2.066e-11, -9.1357e-11, -3.0881e-12, -4.2085e-13,  1.2413e-11, -3.5535e-08, 0,  1.3041e-07],
                         [   2.066e-11,  2.4238e-11, -4.8871e-11, -2.3602e-12,  2.1508e-13, -4.9476e-12, -2.2177e-08, 0,  8.7873e-08],
                         [ -9.1357e-11, -4.8871e-11,  3.4409e-10,   1.042e-11,  2.0414e-11, -2.6359e-11,  9.2838e-08, 0, -3.1272e-07],
                         [ -3.0881e-12, -2.3602e-12,   1.042e-11,  8.0458e-13,  9.9356e-14, -1.2273e-13,  5.4482e-09, 0, -1.2461e-08],
                         [ -4.2085e-13,  2.1508e-13,  2.0414e-11,  9.9356e-14,  1.9856e-11, -1.2714e-11, -1.6628e-09, 0, -1.9713e-08],
                         [  1.2413e-11, -4.9476e-12, -2.6359e-11, -1.2273e-13, -1.2714e-11,  3.6357e-11, -1.7573e-08, 0,  6.0969e-08],
                         [ -3.5535e-08, -2.2177e-08,  9.2838e-08,  5.4482e-09, -1.6628e-09, -1.7573e-08,  0.00036231, 0, -0.00014813],
                         [  0         ,  0         ,  0         ,  0         ,  0         ,  0         ,  0         , 0,  0         ],
                         [  1.3041e-07,  8.7873e-08, -3.1272e-07, -1.2461e-08, -1.9713e-08,  6.0969e-08, -0.00014813, 0,  0.0011749 ]])
    }
    return data

def print_elems_out_of_tol(C, Ctruth, relTol, function_name):
    rows, cols = C.shape
    diffsFound = False
    for r in range(rows):
        for c in range(cols):
            if not math.isclose(C[r,c],Ctruth[r,c],rel_tol=relTol):
                if not diffsFound:
                    print(f"Differences found in {function_name}:")
                    diffsFound = True
                print(f" ({r},{c}): calc={C[r,c]} truth={Ctruth[r,c]}")


def test_ric2eci_bad_args():
    validVec = np.array([1, 2, 3])
    invalidVec = np.array([1, 2])
    r3 = np.random.rand(3, 3)
    r3 = (r3 + r3.T) / 2
    
    with pytest.raises(ValueError):
        ct.ric2eci(validVec, validVec, validVec)
    with pytest.raises(ValueError):
        ct.ric2eci(r3, invalidVec, invalidVec)
    with pytest.raises(ValueError):
        ct.ric2eci(r3, validVec, invalidVec)

def test_ric2eci_test3x3(common_test_data):
    relTol = common_test_data["relTol"]
    r = common_test_data["r"]
    v = common_test_data["v"]
    ric_full = common_test_data["ric"]
    eci_full = common_test_data["eci"]
    ric = ric_full[0:3, 0:3]
    eciTruth = eci_full[0:3, 0:3]
    
    eci = ct.ric2eci(ric, r, v)
    print_elems_out_of_tol(eci, eciTruth, relTol, inspect.currentframe().f_code.co_name)
    npt.assert_allclose(eci, eciTruth, rtol=relTol)
    assert issymmetric(eci)

def test_ric2eci_test6x6(common_test_data):
    relTol = common_test_data["relTol"]
    r = common_test_data["r"]
    v = common_test_data["v"]
    ric_full = common_test_data["ric"]
    eci_full = common_test_data["eci"]
    ric = ric_full[0:6, 0:6]
    eciTruth = eci_full[0:6, 0:6]
    
    eci = ct.ric2eci(ric, r, v)
    print_elems_out_of_tol(eci, eciTruth, relTol, inspect.currentframe().f_code.co_name)
    npt.assert_allclose(eci, eciTruth, rtol=relTol)
    assert issymmetric(eci)


def test_ric2eci_test7x7(common_test_data):
    relTol = common_test_data["relTol"]
    r = common_test_data["r"]
    v = common_test_data["v"]
    ric_full = common_test_data["ric"]
    eci_full = common_test_data["eci"]
    ric = ric_full[0:7, 0:7]
    eciTruth = eci_full[0:7, 0:7]
    
    eci = ct.ric2eci(ric, r, v)
    print_elems_out_of_tol(eci, eciTruth, relTol, inspect.currentframe().f_code.co_name)
    npt.assert_allclose(eci, eciTruth, rtol=relTol)
    assert issymmetric(eci)

def test_ric2eci_test8x8(common_test_data):
    relTol = common_test_data["relTol"]
    r = common_test_data["r"]
    v = common_test_data["v"]
    ric_full = common_test_data["ric"]
    eci_full = common_test_data["eci"]
    ric = ric_full[0:8, 0:8]
    eciTruth = eci_full[0:8, 0:8]
    
    eci = ct.ric2eci(ric, r, v)
    print_elems_out_of_tol(eci, eciTruth, relTol, inspect.currentframe().f_code.co_name)
    npt.assert_allclose(eci, eciTruth, rtol=relTol)
    assert issymmetric(eci)

def test_ric2eci_test9x9(common_test_data):
    relTol = common_test_data["relTol"]
    r = common_test_data["r"]
    v = common_test_data["v"]
    ric_full = common_test_data["ric"]
    eci_full = common_test_data["eci"]
    ric = ric_full[0:9, 0:9]
    eciTruth = eci_full[0:9, 0:9]
    
    eci = ct.ric2eci(ric, r, v)
    print_elems_out_of_tol(eci, eciTruth, relTol, inspect.currentframe().f_code.co_name)
    npt.assert_allclose(eci, eciTruth, rtol=relTol)
    assert issymmetric(eci)

def test_expand_transmatrix_not3x3():
    invalidMatrix = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError):
        ct.expand_transmatrix(invalidMatrix, 6)

def test_expand_transmatrix_invalid_len():
    mat = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    with pytest.raises(ValueError):
        ct.expand_transmatrix(mat, 5)

def check_expand_transmatrix_output(mat, outMat, expandSize):
    # Check size of output matrix
    outSize = np.shape(outMat)
    assert len(outSize) == 2
    assert outSize[0] == expandSize
    assert outSize[1] == expandSize
        
    # Check upper left 3x3
    for i in range(3):
        for j in range(3):
            assert outMat[i,j] == mat[i,j]
            
    # Check upper right and lower left 3x3s
    for i in range(3):
        for j in range(3, 6):
            assert outMat[i,j] == 0.0
            assert outMat[j,i] == 0.0
            
    # Check lower right 3x3
    for i in range(3, 6):
        for j in range(3, 6):
            assert outMat[i,j] == mat[i-3,j-3]
            
    numExtra = expandSize - 6
    if numExtra > 0:
        # Check off diagonals for extra elements
        for i in range(6):
            for j in range(6, numExtra):
                assert outMat[i,j] == 0.0
                assert outMat[j,i] == 0.0
    
        # Check diagonal for identitiy matrix
        eye = np.eye(numExtra)
        for i in range(6, numExtra):
            for j in range(6, numExtra):
                assert outMat[i,j] == eye[i-numExtra,j-numExtra]

def test_expand_transmatrix_to6x6():
    mat = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    expandSize = 6
    
    outMat = ct.expand_transmatrix(mat, expandSize)
    check_expand_transmatrix_output(mat, outMat, expandSize)

def test_expand_transmatrix_to7x7():
    mat = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    expandSize = 7
    
    outMat = ct.expand_transmatrix(mat, expandSize)
    check_expand_transmatrix_output(mat, outMat, expandSize)

def test_expand_transmatrix_to8x8():
    mat = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    expandSize = 8
    
    outMat = ct.expand_transmatrix(mat, expandSize)
    check_expand_transmatrix_output(mat, outMat, expandSize)

def test_expand_transmatrix_to9x9():
    mat = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    expandSize = 9
    
    outMat = ct.expand_transmatrix(mat, expandSize)
    check_expand_transmatrix_output(mat, outMat, expandSize)

