"""Provides the Pc Circle 2D probability of collision calculation.
"""

import numpy as np
from warnings import warn
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
    VectorType,
    valid_cov_matrix,
    valid_vector,
    )

def pc_circle(r1: VectorType, v1: VectorType, C1: MatrixType,
              r2: VectorType, v2: VectorType, C2: MatrixType,
              hbr: float, params: dict = {}) -> tuple[float, dict]:
    """Computes Pc for state/cov input by integrating over a circle on
    the conjunction plane.
    
    TODO: Need to fill in with details once implemented

    Parameters
    ----------
    r1 : VectorType
        Primary object's position vector in inertial cartesian
        coordinates, size is 1x3.
    v1 : VectorType
        Primary object's velocity vector in inertial cartesian
        coordinates, size is 1x3.
    C1 : MatrixType
        Primary object's covariance matrix in the same inertial
        cartesian coordinate frame as the position and velocity, size is
        3x3, 6x6, or nxn with n > 6.
    r2 : VectorType
        Secondary object's position vector in inertial cartesian
        coordinates, size is 1x3.
    v2 : VectorType
        Secondary object's velocity vector in inertial cartesian
        coordinates, size is 1x3.
    C2 : MatrixType
        Secondary object's covariance matrix in the same inertial
        cartesian coordinate frame as the position and velocity, size is
        3x3, 6x6, or nxn with n > 6.
    hbr : float
        Coimbined hard body radius of the primary and secondary objects.
    params : dict, optional
        TODO Fill this section out if needed
        _description_, by default {}

    Returns
    -------
    tuple[float, dict]
        Pc - Computed probability of collision
        out - Dictionary containing the following supplemental
              information:
          IsPosDef - Flag indicating if the combined and marginalized
                     covariance has a negative eigenvalue.
          IsRemediated - Flag indicating if the combined and
                         marginalized 2x2 covariance was remediated,
                         either successfully or not.
          Amat - Combined covariance projected onto the nominal
                 conjunction plane.
          xm, zm - Position of the mean relative miss distance on the
                   conjunction plane.
          sx, sz - Sigma values of the relative miss distance PDF on the
                   conjunction plane.
          r1,v1,C1,r2,v2,C2,hbr - Input parameters saved off for use in
                                  other functions.
    
    Raises
    ------
    ValueError
        Occurs when invalid covariance matrix or vectors are passed in.
    
    References
    ----------
    Alfano, S. "A Numerical Implementation of Spherical Object Collision
    Probability." Journal of the Astronautical Sciences, Vol. 53, No. 1,
    pp. 103-109, Jan-Mar 2005.
    """
    
    # Check for valid vectors and matrices
    if not valid_vector(r1):
        raise ValueError("r1 vector must be 1x3 NDarray")
    if not valid_vector(v1):
        raise ValueError("v1 vector must be 1x3 NDarray")
    if not valid_cov_matrix(C1):
        raise ValueError("C1 matrix must be 3x3, 6x6, or nxn (n>6)"
                         + " symmetric NDarray")
    if not valid_vector(r2):
        raise ValueError("r2 vector must be 1x3 NDarray")
    if not valid_vector(v2):
        raise ValueError("v2 vector must be 1x3 NDarray")
    if not valid_cov_matrix(C2):
        raise ValueError("C2 matrix must be 3x3, 6x6, or nxn (n>6)"
                         + " symmetric NDarray")
    c1_size = np.size(C1)
    c2_size = np.size(C2)
    if c1_size != c2_size:
        raise ValueError("C1 and C2 matrices must be the same size")
    
    # Set defaults within params structure
    if "EstimationMode" not in params:
        params["EstimationMode"] = 64
    estimation_mode = params["EstimationMode"]
    
    if "WarningLevel" not in params:
        params["WarningLevel"] = 0
    warning_level = params["WarningLevel"]
    
    if "PriSecCovProcessing" not in params:
        params["PriSecCovProcessing"] = False
    
    # Check for valid and sensible estimation_mode
    if estimation_mode <= 0:
        if estimation_mode != 0 and estimation_mode != -1:
            raise ValueError("Invalid EstimationMode")
    else:
        if not isinstance(estimation_mode, int):
            raise ValueError("Invalid EstimationMode")
        elif estimation_mode < 16 and warning_level > 0:
            warn("EstimationMode specifies fewer than 16 quadrature points, which can cause inaccurate Pc estimates")
    
    
    
    # Save the input parameters into the output structure
    out = {}
    out["r1"] = r1
    out["v1"] = v1
    out["C1"] = C1
    out["r2"] = r2
    out["v2"] = v2
    out["C2"] = C2
    out["hbr"] = hbr
    
    # Combine the covariances
    comb_cov = C1 + C2
    
    # Relative position and velocity
    r = r1 - r2
    v = v1 - v2
    
    # Check and adjust for zero miss distance (for processing Alfano
    # 2009 test cases)
    rmag = np.linalg.norm(r)
    reps = max(10 * np.spacing(rmag), 1.0e-6*hbr)
    if rmag < reps:
        rsum = r1 + r2
        rsum_mag = np.linalg.norm(rsum)
        vmag = np.linalg.norm(v)
        rdel = reps * np.cross(rsum,v) / rsum_mag / vmag
        r = r + rdel
    
    # Check for zero relative velocity (for processing Alfano 2009 test
    # cases)
    vmag = np.linalg.norm(v)
    if vmag == 0:
        # TODO display warning for zero rel vel and set Pc to NaN
        dummy = 1
    
    # Orbit normal
    h = np.cross(r,v)
    
    # Construct the relative encounter frame
    y = v / np.linalg.norm(v)
    z = h / np.linalg.norm(h)
    x = np.cross(y, z)
    eci2xyz = np.stack((x, y, z), axis = 1)
    out["xhat"] = x
    out["yhat"] = y
    out["zhat"] = z

if __name__ == "__main__":
    expPc = 1.807363058494765e-01
            
    r1 = np.array([-3239.128337196251,   2404.575152356222,   5703.228541709001])
    v1 = np.array([-3.745768373154199,   5.012339015927846,  -4.231864565717194])
    r2 = np.array([-3239.138264917246,   2404.568320465936,   5703.235605231182])
    v2 = np.array([ 6.110192790100711,  -1.767321407894830,   4.140369261741708])
    cov1 = np.array([[ 0.342072996423899, -0.412677096778269,  0.371500417511149],
                     [-0.412677096778269,  0.609905946319294, -0.540401385544286],
                     [ 0.371500417511149, -0.540401385544286,  0.521238634755377]])*1e-3
    cov2 = np.array([[ 0.028351300975134, -0.008204103437377,  0.019253747960155],
                     [-0.008204103437377,  0.002404377774847, -0.005586512197914],
                     [ 0.019253747960155, -0.005586512197914,  0.013289250260317]])
    HBR = 0.020
    pc_circle(r1,v1,cov1,r2,v2,cov2,HBR)