"""Augmented math module

Contains some general purpose math utilities for the cara_analysis_tools
library.
"""

import numpy as np
from scipy.linalg import issymmetric
from scipy.special import erf, erfc
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
    VectorType
    )

def cov_make_symmetric(C: MatrixType) -> MatrixType:
    """Makes a covariance matrix diagonlly symmetric

    Parameters
    ----------
    C : MatrixType
        Covariance matrix, must be nxn with n >= 2

    Returns
    -------
    MatrixType
        Symmetrized version of the covariance matrix, nxn
    
    Raises
    ------
    ValueError
        Occurs when a non-square matrix that isn't 2x2 or more is passed
        in.
    """
    
    # Check the data type
    if not isinstance(C, np.ndarray):
        raise ValueError("Matrix must be a numpy array!")
        
    # Check matrix size
    cov_size = np.shape(C)
    if len(cov_size) != 2:
        raise ValueError("Matrix must be a 2D numpy array")
    if cov_size[0] != cov_size[1]:
        raise ValueError("Matrix must be an nxn 2D numpy array")
    if cov_size[0] < 2:
        raise ValueError("Matrix must be an nxn 2D numpy array with n >= 2")
    
    if issymmetric(C):
        # Don't change anything if the matrix is already symmetric
        Csym = C
    else:
        Ct = C.T
        # Average out any off-diagonal asymmetries
        Csym = (C + Ct) / 2
        # Reflect about the diagonal to ensure diagonal symmetry absolutely
        Csym = np.triu(Csym) + np.triu(Csym, k=1).T
    
    return Csym

def erf_vec_dif(a: VectorType, b: VectorType) -> VectorType:
    """
    Calculate the difference d = erf(a) - erf(b), using erfc for cases of
    large positive or negative values of a and b to provide improved accuracy.

    Parameters
    ----------
    a : VectorType
        First argument. Must have the same shape as b.
    b : VectorType
        Second argument. Must have the same shape as a.

    Returns
    -------
    d : VectorType
        Element-wise difference erf(a) - erf(b), computed with improved
        numerical accuracy for large-magnitude inputs.
    """
    
    # Stack a and b column-wise to find element-wise min/max
    ab = np.stack([a.ravel(), b.ravel()], axis=1)
    minab = ab.min(axis=1).reshape(a.shape)
    maxab = ab.max(axis=1).reshape(a.shape)

    # Threshold for switching to erfc for better accuracy
    large = 3.0

    # Initialize output array
    d = np.full(a.shape, np.nan)

    # Large positive a & b: use erfc for better accuracy
    # erf(a) - erf(b) = (1 - erfc(a)) - (1 - erfc(b)) = erfc(b) - erfc(a)
    set1 = minab > large
    d[set1] = erfc(b[set1]) - erfc(a[set1])

    # Large negative a & b: use erfc with negated arguments
    # erf(x) = -erf(-x), so erf(a) - erf(b) = erfc(-a) - erfc(-b)
    set2 = maxab < -large
    d[set2] = erfc(-a[set2]) - erfc(-b[set2])

    # All other cases: direct erf computation
    set3 = ~set1 & ~set2
    d[set3] = erf(a[set3]) - erf(b[set3])

    return d
