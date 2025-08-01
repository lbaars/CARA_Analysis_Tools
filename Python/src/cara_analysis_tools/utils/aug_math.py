"""Augmented math module

Contains some general purpose math utilities for the cara_analysis_tools
library.
"""

import numpy as np
from scipy.linalg import issymmetric
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
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

