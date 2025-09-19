"""Provides the Pc Circle 2D probability of collision calculation.
"""

import numpy as np
from warnings import warn
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
    VectorType,
    )
import cara_analysis_tools.pc.utils as pcu

def pc_circle(r1: VectorType, v1: VectorType, cov1: MatrixType,
              r2: VectorType, v2: VectorType, cov2: MatrixType,
              hbr: float | VectorType, params: dict = {}) -> tuple[float, dict]:
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
    cov1 : MatrixType
        Primary object's covariance matrix in the same inertial
        cartesian coordinate frame as the position and velocity, size is
        3x3, 6x6, or nxn with n > 6.
    r2 : VectorType
        Secondary object's position vector in inertial cartesian
        coordinates, size is 1x3.
    v2 : VectorType
        Secondary object's velocity vector in inertial cartesian
        coordinates, size is 1x3.
    cov2 : MatrixType
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
    
    # Set defaults within params structure
    if "EstimationMode" not in params:
        params["EstimationMode"] = 64
    estimation_mode = params["EstimationMode"]
    
    if "WarningLevel" not in params:
        params["WarningLevel"] = 0
    warning_level = params["WarningLevel"]
    
    # Enables plotting of primary and secondary ellipses on CA
    # distribution plots
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
    
    # Reformat the inputs to expected dimensions
    #  (nx3 for pos and vel; nx9 for covariances, nx1 for HBRs)
    (n_vec,  r1, v1) = pcu.check_and_resize_posvel(r1, v1)
    (n_vec2, r2, v2) = pcu.check_and_resize_posvel(r2, v2)
    if (n_vec != n_vec2):
        raise ValueError("Number of primary and secondary postions must be equal")
    cov1 = pcu.check_and_resize_cov(n_vec,  cov1)
    cov2 = pcu.check_and_resize_cov(n_vec2, cov2)
    
    # Replicate scalar HBR into an nx1 array
    if isinstance(hbr, float):
        hbr = np.tile(hbr, (n_vec, 1))
    elif isinstance(hbr, np.ndarray):
        hbr_size = np.shape(hbr)
        if len(hbr_size) == 1:
            if hbr_size[0] == 1:
                hbr = np.tile(hbr[0], (n_vec, 1))
            elif hbr_size[0] == n_vec:
                hbr = np.array([hbr])
            else:
                raise ValueError("Size of hbr array must be 1x1 or nx1")
        elif len(hbr_size) == 2:
            if (hbr_size[0] != 1 or hbr_size[1] != 1) and \
                (hbr_size[0] != n_vec or hbr_size[1] != 1):
                    raise ValueError("Size of hbr array must be 1x1 or nx1")
        else:
            raise ValueError("Size of hbr array must be 1x1 or nx1")
    else:
        raise ValueError("hbr parameter must be a float or numpy array")
    
    # Ensure HBR values are nonnegative
    if (hbr < 0).any():
        if warning_level > 0:
            warn("Negative HBR values found and replaced with zeros")
        hbr[hbr < 0] = 0

    # Save the input parameters into the output structure
    out = {}
    out["r1"] = r1
    out["v1"] = v1
    out["cov1"] = cov1
    out["r2"] = r2
    out["v2"] = v2
    out["cov2"] = cov2
    out["hbr"] = hbr
    
    # Combine the covariances
    comb_cov = cov1 + cov2
    
    # Relative position and velocity
    r = r1 - r2
    v = v1 - v2
    
    # Check and adjust for zero miss distance (for processing Alfano
    # 2009 test cases)
    rmag = np.sqrt(r[:,0]**2 + r[:,1]**2 + r[:,2]**2)
    reps = np.maximum(10.0 * np.spacing(rmag), (1.0e-6 * hbr).flatten())
    small_rmag = (rmag < reps)
    if np.sum(small_rmag) > 0:
        if warning_level > 0:
            warn("Zero or near-zero miss distance cases found; perturbing miss distance for those cases")
        rsum = r1[small_rmag,:] + r2[small_rmag,:]
        rsum_mag = np.sqrt(rsum[:,0]**2 + rsum[:,1]**2 + rsum[:,2]**2)
        vmag = np.sqrt(v[small_rmag,0]**2 + v[small_rmag,1]**2 + v[small_rmag,2]**2)
        reps2 = reps[small_rmag].reshape(-1, 1)
        cross_prod = np.cross(rsum,v[small_rmag,:])
        rdel = reps2 * cross_prod / rsum_mag.reshape(-1, 1) / vmag.reshape(-1, 1)
        r[small_rmag,:] = r[small_rmag,:] + rdel
    
    # Check for zero relative velocity (for processing Alfano 2009 test
    # cases)
    #TODO: continue implementation from here
    vmag = np.linalg.norm(v)
    if vmag == 0:
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
            
    r1 = np.array([[-3239.128337196251,   2404.575152356222,   5703.228541709001],
                   [-3239.128337196251,   2404.575152356222,   5703.228541709001],
                   [-3239.128337196251,   2404.575152356222,   5703.228541709001],
                   [-3239.128337196251,   2404.575152356222,   5703.228541709001]])
    v1 = np.array([[-3.745768373154199,   5.012339015927846,  -4.231864565717194],
                   [-3.745768373154199,   5.012339015927846,  -4.231864565717194],
                   [-3.745768373154199,   5.012339015927846,  -4.231864565717194],
                   [-3.745768373154199,   5.012339015927846,  -4.231864565717194]])
    r2 = np.array([[-3239.138264917246,   2404.568320465936,   5703.235605231182],
                   [-3239.128337196251,   2404.575152356222,   5703.228541709001],
                   [-3239.138264917246,   2404.568320465936,   5703.235605231182],
                   [-3239.128337196251,   2404.575152356222,   5703.228541709001]])
    v2 = np.array([[ 6.110192790100711,  -1.767321407894830,   4.140369261741708],
                   [ 6.110192790100711,  -1.767321407894830,   4.140369261741708],
                   [ 6.110192790100711,  -1.767321407894830,   4.140369261741708],
                   [ 6.110192790100711,  -1.767321407894830,   4.140369261741708],])
    cov1 = np.array([[ 0.342072996423899, -0.412677096778269,  0.371500417511149],
                     [-0.412677096778269,  0.609905946319294, -0.540401385544286],
                     [ 0.371500417511149, -0.540401385544286,  0.521238634755377]])*1e-3
    cov2 = np.array([[ 0.028351300975134, -0.008204103437377,  0.019253747960155],
                     [-0.008204103437377,  0.002404377774847, -0.005586512197914],
                     [ 0.019253747960155, -0.005586512197914,  0.013289250260317]])
    HBR = 0.020
    pc_circle(r1,v1,cov1,r2,v2,cov2,HBR)