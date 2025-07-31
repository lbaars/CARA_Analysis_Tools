import numpy as np
import cara_analysis_tools.utils.datatypes as dt

def test_valid_cov_matrix():
    notNumpyArray = list[1., 2.]
    notMatrix = np.array([1, 2, 3])
    notSquare = np.array([[1, 2, 3], [4, 5, 6]])
    notSymmetric = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    oneByOne = np.array([[1]])
    r2 = np.random.rand(2, 2)
    twoByTwo = (r2 + r2.T) / 2
    r3 = np.random.rand(3, 3)
    threeByThree = (r3 + r3.T) / 2
    r4 = np.random.rand(4, 4)
    fourByFour = (r4 + r4.T) / 2
    r5 = np.random.rand(5, 5)
    fiveByFive = (r5 + r5.T) / 2
    r6 = np.random.rand(6, 6)
    sixBySix = (r6 + r6.T) / 2
    r7 = np.random.rand(7, 7)
    sevenBySeven = (r7 + r7.T) / 2
    r8 = np.random.rand(8, 8)
    eightByEight = (r8 + r8.T) / 2
    
    # Test failure cases
    assert not dt.valid_cov_matrix(notNumpyArray)
    assert not dt.valid_cov_matrix(notMatrix)
    assert not dt.valid_cov_matrix(notSquare)
    assert not dt.valid_cov_matrix(notSymmetric)
    assert not dt.valid_cov_matrix(oneByOne)
    assert not dt.valid_cov_matrix(twoByTwo)
    assert not dt.valid_cov_matrix(fourByFour)
    assert not dt.valid_cov_matrix(fiveByFive)
    
    # Test success cases
    assert dt.valid_cov_matrix(threeByThree)
    assert dt.valid_cov_matrix(sixBySix)
    assert dt.valid_cov_matrix(sevenBySeven)
    assert dt.valid_cov_matrix(eightByEight)

def test_valid_3x3_matrix():
    notNumpyArray = list[1., 2.]
    notMatrix = np.array([1, 2, 3])
    notSquare = np.array([[1, 2, 3], [4, 5, 6]])
    r1 = np.array([[1]])
    r2 = np.random.rand(2, 2)
    r3 = np.random.rand(3, 3)
    r4 = np.random.rand(4, 4)
    
    # Test failure cases
    assert not dt.valid_3x3_matrix(notNumpyArray)
    assert not dt.valid_3x3_matrix(notMatrix)
    assert not dt.valid_3x3_matrix(notSquare)
    assert not dt.valid_3x3_matrix(r1)
    assert not dt.valid_3x3_matrix(r2)
    assert not dt.valid_3x3_matrix(r4)
    
    # Test success cases
    assert dt.valid_3x3_matrix(r3)
    

def test_valid_vector():
    notNumpyArray = list[1., 2.]
    notVector = np.array([[1, 2, 3],[4, 5, 6]])
    size1 = np.array([1])
    size2 = np.array([1, 2])
    size3 = np.array([1, 2, 3])
    size4 = np.array([1, 2, 3, 4])
    size5 = np.array([1, 2, 3, 4, 5])
    size6 = np.array([1, 2, 3, 4, 5, 6])
    
    # Test failure cases
    assert not dt.valid_vector(notNumpyArray)
    assert not dt.valid_vector(notVector)
    assert not dt.valid_vector(size1)
    assert not dt.valid_vector(size2)
    assert not dt.valid_vector(size4)
    assert not dt.valid_vector(size5)
    assert not dt.valid_vector(size6)
    
    # Test success cases
    assert dt.valid_vector(size3)

