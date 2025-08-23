"""Provides helper utilities for Probability of Collision algorithms.
"""

import numpy as np
from typing import Tuple
from cara_analysis_tools.utils.datatypes import MatrixType

def product3x3(a: MatrixType, b: MatrixType) -> MatrixType:
    """Vectorized 3x3 matrix multiplication routine
    
    Input and output matrices are nx9 2D numpy arrays representing n
    separate matrix multiplications to perform. Each row represents a
    matrix in the form of:
     [x[0,0], x[0,1], x[0,2], x[1,0], x[1,1], x[1,2], x[2,0], x[2,1], x[2,2] ]
    
    Parameters
    ----------
    a : MatrixType
        Left matrix in the computation, size nx9 (n>0)
    b : MatrixType
        Right matrix in the computation, size nx9 (n>0)

    Returns
    -------
    MatrixType
        Matrix product, size nx9 (n>0)
    
    Raises
    ------
    ValueError
        Occurs when the input matrices are not 2D numpy arrays with 9
        columns or if the matrices are not the same size.
    """
    
    # Check the data type
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("Input arrays must be numpy arrays")
        
    # Check matrix size
    a_size = np.shape(a)
    b_size = np.shape(b)
    if len(a_size) != 2 or len(b_size) != 2:
        raise ValueError("Input arrays must be 2D numpy arrays")
    if a_size[1] != 9 or b_size[1] != 9:
        raise ValueError("Input arrays must each have 9 columns")
    if a_size[0] != b_size[0]:
        raise ValueError("Input arrays must have the same number of rows")
    
    # Calculate the product
    out = np.full(shape = a_size, fill_value=np.nan)
    out[:,0] = a[:,0]*b[:,0] + a[:,1]*b[:,3] + a[:,2]*b[:,6]
    out[:,1] = a[:,0]*b[:,1] + a[:,1]*b[:,4] + a[:,2]*b[:,7]
    out[:,2] = a[:,0]*b[:,2] + a[:,1]*b[:,5] + a[:,2]*b[:,8]
    out[:,3] = a[:,3]*b[:,0] + a[:,4]*b[:,3] + a[:,5]*b[:,6]
    out[:,4] = a[:,3]*b[:,1] + a[:,4]*b[:,4] + a[:,5]*b[:,7]
    out[:,5] = a[:,3]*b[:,2] + a[:,4]*b[:,5] + a[:,5]*b[:,8]
    out[:,6] = a[:,6]*b[:,0] + a[:,7]*b[:,3] + a[:,8]*b[:,6]
    out[:,7] = a[:,6]*b[:,1] + a[:,7]*b[:,4] + a[:,8]*b[:,7]
    out[:,8] = a[:,6]*b[:,2] + a[:,7]*b[:,5] + a[:,8]*b[:,8]
    
    return out

def check_and_resize_nx3(a: MatrixType) -> MatrixType:
    """Resizes the input array into a 2D nx3 array.

    Parameters
    ----------
    a : MatrixType
        1D or 2D numpy array. If a 1D array, it must be of size 3. If a
        2D array, must be of size nx3.

    Returns
    -------
    MatrixType
        A 2D numpy array of size nx3.
    
    Raises
    ------
    ValueError
        Occurs when the input matrices are not numpy arrays or are of
        incorrect sizes.
    """
    
    # Check the data type
    if not isinstance(a, np.ndarray):
        raise ValueError("Input array must be numpy array")
    
    a_size = np.shape(a)
    if len(a_size) == 1:
        if a_size[0] != 3:
            raise ValueError("Array must have a size of 3 or nx3")
        a = a.reshape(1,3)
        a_size = np.shape(a)
    if len(a_size) != 2:
        raise ValueError("Array must have a size of 3 or nx3")
    if a_size[1] != 3:
        raise ValueError("Array must have a size of 3 or nx3")
    
    return a

def check_and_resize_posvel(r: MatrixType, v: MatrixType) -> \
                            Tuple[int, MatrixType, MatrixType]:
    """Checks the input vectors as valid representations of position and
    velocity.
    
    Position and velocity information should be represented as 1D arrays
    with 3 elements each or as 2D nx3 arrays. If the r array is a nx3
    with n>1 and the v array is a 1D array or a 2D array of size 1x3,
    then the v array will be repeated n times and returned as an output
    2D array of size nx3.

    Parameters
    ----------
    r : MatrixType
        1D or 2D numpy array with position information. If a 1D array,
        must have a size of 3. If a 2D array, must be of size nx3.
    v : MatrixType
        1D or 2D numpy array with velocity information. If a 1D array,
        must have a size of 3. If a 2D array, must be of size nx3.

    Returns
    -------
    Tuple[int, MatrixType, MatrixType]
        num_r (int) - Number of rows of the input r.
        r (MatrixType) - Input r, converted to a 2D numpy array if a 1D
                         array was passed in.
        v (MatrixType) - Input v, converted to a 2D numpy array if a 1D
                         array was passed in. If r was a 2D array of
                         size nx3 and v had only one row, then v is
                         repeated n times to create a nx3 output.
    
    Raises
    ------
    ValueError
        Occurs when the input matrices are not numpy arrays or are of
        incorrect sizes or if the v array cannot be properly resized.
    """
    
    try:
        r = check_and_resize_nx3(r)
    except ValueError:
        raise ValueError("r array must have a size of 3 or nx3")
    try:
        v = check_and_resize_nx3(v)
    except ValueError:
        raise ValueError("v array must have a size of 3 or nx3")
    
    # Resize v vector, if needed
    r_size = np.shape(r)
    v_size = np.shape(v)
    num_r = r_size[0]
    num_v = v_size[0]
    if num_r != num_v:
        if num_v == 1:
            v = np.tile(v, (num_r, 1))
        else:
            raise ValueError("v matrix cannot be resized to match r matrix")
    
    return num_r, r, v

# TODO: Need to implement check_and_resize_cov

