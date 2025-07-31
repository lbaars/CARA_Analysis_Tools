"""Provides covariance transformations between coordinate frames

Using the inertial position and velocity passed in, the functions in
this module will transform covariance matrices between coordinate frames
such as RIC, ECI, etc.

"""

import numpy as np
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
    VectorType,
    valid_cov_matrix,
    valid_3x3_matrix,
    valid_vector,
    )
from cara_analysis_tools.utils.aug_math import cov_make_symmetric

#TODO remove pudb import
#import pudb
#pudb.set_trace()

def ric2eci(ric: MatrixType,
            r: VectorType,
            v: VectorType,
            make_symmetric: bool = True) -> MatrixType:
    """Transforms input vector or matrix from the RIC to the ECI frame.

    Parameters
    ----------
    ric : MatrixType
        Covariance matrix in RIC frame, size is 3x3, 6x6, or nxn with
        n > 6.
    r : VectorType
        ECI position vector in km, size is 1x3.
    v : VectorType
        ECI velocity vector in km/sec, size is 1x3.
    make_symmetric : bool, optional
        _description_, by default True

    Returns
    -------
    MatrixType
        Covariance matrix in ECI frame, size is the same as the input
        ric matrix.
    
    Raises
    ------
    ValueError
        Occurs when invalid covariance matrix or vectors are passed in.
    """
    
    # Check for valid input types
    if not valid_cov_matrix(ric):
        raise ValueError("ric matrix must be 3x3, 6x6, or nxn (n>6)"
                         + " symmetric NDarray")
    if not valid_vector(r):
        raise ValueError("r vector must be 1x3 NDarray")
    if not valid_vector(v):
        raise ValueError("v vector must be 1x3 NDarray")
    
    # Setting up vectors in the radial, in-track, and cross-track directions
    h = np.cross(r, v)
    rhat = r / np.linalg.norm(r)
    chat = h / np.linalg.norm(h)
    ihat = np.cross(chat, rhat)
    
    # Create rotation matrix for 3x3 covariance
    RICtoECI = np.stack((rhat, ihat, chat), axis = 1)
    
    # Determine how many additional terms are needed
    cov_size = np.shape(ric)[0]
    if cov_size >= 6:
        RICtoECI = expand_transmatrix(RICtoECI, cov_size)
    
    # Transform the covariance matrix to ECI coordinates
    eci = RICtoECI @ ric @ RICtoECI.T
    
    # Make the covariance symmetric, if needed
    if make_symmetric:
        eci = cov_make_symmetric(eci)
    
    return eci
    
def expand_transmatrix(trans: MatrixType, len: int) -> MatrixType:
    """Expands a transformation matrix to the length passed in
    
    Takes a 3x3 transformation matrix and expands it to 6x6, or nxn with
    n>6. For a 6x6 output matrix, the original transformation matrix is
    repeated in the lower right quadrant while the upper right and lower
    left quadrants are zero filled. For an nxn output matrix, each term
    past 6 will have off-diagonals zero filled and the diagonals filled
    with ones.

    Parameters
    ----------
    trans : MatrixType
        A 3x3 transformation matrix
    len : int
        Size of the expanded matrix, must be 6 or n with n>6.

    Returns
    -------
    MatrixType
        Expanded transformation matrix with dimensions equal to len.
    
    Raises
    ------
    ValueError
        Occurs when the transformation matrix isn't a valid 3x3
        transformation matrix. This exception is also raised if len is
        not greater than or equal to 6.
    """
    
    # Check for valid input types
    if not valid_3x3_matrix(trans):
        raise ValueError("trans matrix must be 3x3 NDarray")
    if len < 6:
        raise ValueError("len must be >= 6")
    
    # Determine the number of additional terms
    additional_terms = len - 6
    
    # Start by creating the 6x6
    zero = np.zeros((3,3))
    topHalf = np.concatenate((trans, zero), axis = 1)
    bottomHalf = np.concatenate((zero, trans), axis = 1)
    trans = np.concatenate((topHalf, bottomHalf), axis = 0)
    
    # Expand the matrix further if needed
    if additional_terms > 0:
        zeroTop = np.zeros((6,additional_terms))
        zeroBottom = np.zeros((additional_terms,6))
        ident = np.eye(additional_terms)
        topHalf = np.concatenate((trans, zeroTop), axis = 1)
        bottomHalf = np.concatenate((zeroBottom, ident), axis = 1)
        trans = np.concatenate((topHalf, bottomHalf), axis = 0)
    
    return trans

