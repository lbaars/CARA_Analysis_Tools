"""Provides helper utilities for Probability of Collision algorithms.
"""

import numpy as np
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

