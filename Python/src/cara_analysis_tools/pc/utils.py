"""Provides helper utilities for Probability of Collision algorithms.
"""

import numpy as np
from typing import Tuple
from cara_analysis_tools.utils.datatypes import MatrixType, VectorType

class PcCalculationError(Exception):
    """Custom exception raised when Pc calculation errors are encountered."""
    pass

def eig2x2(Araw: MatrixType) -> \
        tuple[MatrixType, MatrixType, VectorType, VectorType]:
    """Vectorized eigenvalue and eigenvector solver for 2x2 symmetric matrices
    
    Parameters
    ----------
    Araw : MatrixType
        Array of shape (n, 3) where each row is [a, b, d] representing a
        symmetric matrix:
            [ a  b ]
            [ b  d ]
    
    Returns
    -------
    tuple[MatrixType, MatrixType, VectorType, VectorType]
        V1 - Array (n,2), first eigenvector of each matrix
        V2 - Array (n,2), second eigenvector of each matrix
        L1 - Array (n,), largest eigenvalue
        L2 - Array (n,), smallest eigenvalue
    """

    # split components
    a = Araw[:, 0]
    b = Araw[:, 1]
    d = Araw[:, 2]

    # trace and determinant
    T = a + d
    D = a * d - b * b

    # eigenvalues (quadratic closed form)
    sqrt_term = np.sqrt(np.clip(T * T - 4 * D, 0.0, None))
    L1 = (T + sqrt_term) / 2.0  # largest
    L2 = (T - sqrt_term) / 2.0  # smallest

    # prepare output vectors
    n = Araw.shape[0]
    V1 = np.full((n, 2), np.nan)
    V2 = np.full((n, 2), np.nan)

    # indices where off-diagonal b != 0
    mask = b != 0
    if np.any(mask):
        # formula for eigenvectors when b != 0
        v1 = np.column_stack((L1[mask] - d[mask], b[mask]))
        v2 = np.column_stack((L2[mask] - d[mask], b[mask]))
        # normalize
        V1[mask] = v1 / np.linalg.norm(v1, axis=1)[:, None]
        V2[mask] = v2 / np.linalg.norm(v2, axis=1)[:, None]

    # handle diagonal case (b == 0)
    mask0 = ~mask
    if np.any(mask0):
        # if d <= a → first eigenvector = [1 0], second = [0 1]
        diag_le = d[mask0] <= a[mask0]
        idx_le = np.where(mask0)[0][diag_le]
        idx_gt = np.where(mask0)[0][~diag_le]

        V1[idx_le] = np.array([1.0, 0.0])
        V2[idx_le] = np.array([0.0, 1.0])

        V1[idx_gt] = np.array([0.0, 1.0])
        V2[idx_gt] = np.array([1.0, 0.0])

    # special cases where numerical issues occur
    # i.e., sqrt term underflow or b small → fall back to full eig
    small_b = (np.abs(b) < 1e-2) & (b != 0)
    det_err = (a * d == a * d - b * b) & (b * b != 0)
    fallback = small_b | det_err

    if np.any(fallback):
        for i in np.where(fallback)[0]:
            mat = np.array([[a[i], b[i]],
                            [b[i], d[i]]])
            # compute full eig
            vals, vecs = np.linalg.eig(mat)
            # sort ascending eigenvalues like Matlab's eig(...,'vector')
            order = np.argsort(vals)
            vals_sorted = vals[order]
            vecs_sorted = vecs[:, order]
            # assign outputs such that the largest eigenvalue and
            # associated eigenvector are in L1 and V1 variables,
            # respectively
            L2[i] = vals_sorted[0]
            L1[i] = vals_sorted[1]
            V2[i] = vecs_sorted[:, 0]
            V1[i] = vecs_sorted[:, 1]

    return V1, V2, L1, L2

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

def check_and_resize_cov(num_r: int, cov: MatrixType) -> MatrixType:
    """Resizes the covariance matrices passed in into an nx9 matrix to
    be used in vectorized covariance processing.
    
    Reformats the input covariance matrices into an nx9 matrix
    representing the 3x3 position covariance with each row in the
    following format:
      cov = [C(0,0) C(0,1) C(0,2) C(1,0) C(1,1) C(1,2) C(2,0), C(2,1), C(2,2)]
    The "n" is equal to the num_r passed in.
    
    Depending on the input format, this function will respond in several
    different ways to reformat the cov matrix:
    - If the input is an nx9, then this function verifies that n matches
      num_r.
    - If the input is a 3x3, then this function will reformat the matrix
      into a 1x9 and then repeat this vector num_r times to create n
      rows.
    - If the input is a 6x6, then this function will take the upper left
      3x3 component and reformat it into a 1x9. The it will repeat this
      vector num_r times to create n rows.
    - If the input is a nx3x3, then this function will verify that n
      matches num_r and will reformat each 3x3 into a corresponding 1x9.
    - If the input is a nx6x6, then this function will verify that n
      matches num_r and will convert the upper left 3x3 of each 6x6 into
      a corresponding 1x9.
    - Any other input is considered an error.
    
    Parameters
    ----------
    num_r : int
        Number of rows, n, to create
    cov : MatrixType
        Covariance matrix to convert into an nx9. Allowed inputs
        are:
          1D numpy array with 9 elements
          2D numpy array of size 1x9 or nx9
          2D numpy array of size 3x3
          2D numpy array of size 6x6
          3D numpy array of size nx3x3
          3D numpy array of size nx6x6

    Returns
    -------
    MatrixType
        cov - Covariance matrix converted into nx9 format.
    
    Raises
    ------
    ValueError
        Occurs when the input matrix has an incorrect format and cannot
        be resized.
    """
    
    # Check the data type
    if not isinstance(cov, np.ndarray):
        raise ValueError("cov argument must be numpy array")
    
    cov_size = np.shape(cov)
    # If a 1D array, verify size and convert into 2D array
    if len(cov_size) == 1:
        if cov_size[0] != 9:
            raise ValueError("1D cov array must have a size of 9")
        cov = cov.reshape(1,9)
        cov_size = np.shape(cov)
    
    # Resize a 2D array
    if len(cov_size) == 2:
        if cov_size[1] != 9 and \
            (cov_size[0] != 3 or cov_size[1] != 3) and \
            (cov_size[0] != 6 or cov_size[1] != 6):
            raise ValueError("2D cov array must have 9 columns or " + \
                "be a 3x3 or 6x6 matrix")
        # Resize down to a 3x3 if a 6x6 was passed in
        if cov_size[0] == 6 and cov_size[1] == 6:
            cov = cov[0:3,0:3]
            cov_size = np.shape(cov)
        
        # Convert 1x9 into nx9
        if cov_size[0] == 1:
            cov = np.tile(cov, (num_r, 1))
        elif cov_size[0] == 3 and cov_size[1] == 3:
            cov = cov.reshape(1, 9)
            cov = np.tile(cov, (num_r, 1))
        elif cov_size[0] != num_r:
            raise ValueError("2D cov array cannot be resized to " + \
                "match num_r rows")
    # Resize a 3D array
    elif len(cov_size) == 3:
        if (cov_size[0] != num_r or cov_size[1] != 3 or cov_size[2] != 3) and \
           (cov_size[0] != num_r or cov_size[1] != 6 or cov_size[2] != 6):
            raise ValueError("3D cov array must be of size num_rx3x3 " + \
                "or num_rx6x6")
        # Resize down to num_rx3x3 if num_rx6x6 was passed in
        if cov_size[1] == 6 and cov_size[2] == 6:
            cov = cov[:,0:3,0:3]
        cov = cov.reshape(num_r, 9)
    else:
        raise ValueError("Improperly size cov array was detected")
    
    return cov
