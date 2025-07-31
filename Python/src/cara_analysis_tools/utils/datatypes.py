"""Defines datatypes and associated checker functions used by
cara_analysis_tools
"""

import numpy as np
from scipy.linalg import issymmetric
from typing import TypeAlias
from numpy.typing import NDArray

MatrixType: TypeAlias = NDArray[np.float64]
VectorType: TypeAlias = NDArray[np.float64]

def valid_cov_matrix(cov: MatrixType) -> bool:
    """Checks that the matrix passed in is a covariance matrix

    Parameters
    ----------
    cov : MatrixType
        Matrix to check, valid sizes are 3x3, 6x6, or nxn with n>6. The
        matrix also has to be symmetric.

    Returns
    -------
    bool
        Returns True if a valid covariance matrix is found, returns
        False otherwise
    """
    
    # Check the data type
    if not isinstance(cov, np.ndarray):
        return False
        
    # Check cov size and symmetry
    cov_size = np.shape(cov)
    if len(cov_size) != 2:
        return False
    if cov_size[0] != cov_size[1] or not issymmetric(cov):
        return False
    if cov_size[0] != 3 and cov_size[0] < 6:
        return False
    
    return True


def valid_3x3_matrix(mat: MatrixType) -> bool:
    """Checks that the matrix passed in is a 3x3 matrix

    Parameters
    ----------
    cov : MatrixType
        Matrix to check, valid sizes is 3x3.

    Returns
    -------
    bool
        Returns True if a valid 3x3 matrix is found, returns False
        otherwise
    """
    
    # Check the data type
    if not isinstance(mat, np.ndarray):
        return False
        
    # Check matrix size
    mat_size = np.shape(mat)
    if len(mat_size) != 2:
        return False
    if mat_size[0] != mat_size[1]:
        return False
    if mat_size[0] != 3:
        return False
    
    return True


def valid_vector(vec: VectorType) -> bool:
    """Checks that the parameter passed in is a valid 1x3 vector.

    Parameters
    ----------
    vec : VectorType
        Vector to check, must be a 1x3 NDarray.

    Returns
    -------
    bool
        Returns True for a valid 1x3 NDarray and False for anything
        else.
    """
    
    # Check the data type
    if not isinstance(vec, np.ndarray):
        return False
    
    # Check vec size
    vec_size = np.shape(vec)
    if len(vec_size) != 1:
        return False
    if vec_size[0] != 3:
        return False
    
    return True
