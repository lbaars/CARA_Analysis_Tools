"""Provides the Pc Circle 2D probability of collision calculation.
"""

import numpy as np
from warnings import warn
from cara_analysis_tools.utils.datatypes import (
    MatrixType,
    VectorType,
    )
import cara_analysis_tools.pc.utils as pcu
import cara_analysis_tools.utils.aug_math as am
from cara_analysis_tools.pc.utils import PcCalculationError

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
    
    PcCalculationError
        Occurs when two non-positive eigenvalues are found in the combined
        covariance matrix.
    
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
    vmag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2).reshape(-1, 1)
    zero_vmag = (vmag == 0)
    if np.sum(zero_vmag) > 0 and warning_level > 0:
        warn("Zero relative velocity cases found; setting Pc to NaN for those cases")
    
    # Orbit normal
    h = np.cross(r,v)
    
    # Construct the relative encounter frame
    y = v / np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2).reshape(-1, 1)
    z = h / np.sqrt(h[:,0]**2 + h[:,1]**2 + h[:,2]**2).reshape(-1, 1)
    x = np.cross(y, z)
    eci2xyz = np.concatenate((x, y, z), axis = 1)
    out["xhat"] = x
    out["yhat"] = y
    out["zhat"] = z
    
    # Pri and Sec cov processing
    if params["PriSecCovProcessing"]:
        # Project primary covariances into conjunction plane
        rotated_cov = pcu.product3x3(eci2xyz,pcu.product3x3(cov1[:,0:9],eci2xyz[:,[0, 3, 6, 1, 4, 7, 2, 5, 8]]))
        a_mat = rotated_cov[:, [0, 2, 8]]
        out["AmatPri"] = a_mat
        if np.max(np.abs(a_mat.flatten())) == 0:
            warn("All zero primary covariance being processed")
        
        # Calculate eigenvalues and eigenvectors for primary and save off
        # values
        (V1, V2, L1, L2) = pcu.eig2x2(a_mat)
        out["EigV1Pri"] = V1
        out["EigV2Pri"] = V2
        out["EigL1Pri"] = L1
        out["EigL2Pri"] = L2
        
        # Project secondary coavariance into conjunction plane
        rotated_cov = pcu.product3x3(eci2xyz,pcu.product3x3(cov2[:,0:9],eci2xyz[:,[0, 3, 6, 1, 4, 7, 2, 5, 8]]))
        a_mat = rotated_cov[:, [0, 2, 8]]
        out["AmatSec"] = a_mat
        if np.max(np.abs(a_mat.flatten())) == 0:
            warn("All zero secondary covariance being processed")
        
        # Calculate eigenvalues and eigenvectors for secondary and save off
        # values
        (V1, V2, L1, L2) = pcu.eig2x2(a_mat)
        out["EigV1Sec"] = V1
        out["EigV2Sec"] = V2
        out["EigL1Sec"] = L1
        out["EigL2Sec"] = L2
    
    # Project the combined covariance into the conjunction plane
    rotated_cov = pcu.product3x3(eci2xyz,pcu.product3x3(comb_cov[:,0:9],eci2xyz[:,[0, 3, 6, 1, 4, 7, 2, 5, 8]]))
    a_mat = rotated_cov[:, [0, 2, 8]]
    out["Amat"] = a_mat
    
    # Calculate eigenvalues and eigenvectors for the combined covariance, the
    # 2nd eigenvector isn't needed for the rest of the calculation
    (V1, V2, L1, L2) = pcu.eig2x2(a_mat)
    out["EigV1"] = V1
    out["EigV2"] = V2
    out["EigL1"] = L1
    out["EigL2"] = L2
    
    # Issue error if any cases are found with two non-positive eigenvalues
    if (L1 <= 0).any():
        raise PcCalculationError("Invalid case(s) found with two non-positive eigenvalues")
    
    # Issue a warning for any NPD cases
    if warning_level > 0 and (L2 <= 0).any():
        warn("NPD covariance(s) found; remediating using eigenvalue clipping method")
    
    # Use eigenvalue clipping method to make the covariances positive
    # definite. Since this Pc algorithm does not require Cholesky
    # factorization, the covariance remediation is very simple: clip any
    # eigenvalues that are less than the clipping limit and then
    # recreate the remediated covariance using the origianl eigenvectors
    # and the clipped eigenvalues.
    a = np.isinf(hbr)
    finite_hbr = np.logical_not(np.isinf(hbr))
    f_clip = 1.0e-4
    l_rem = (f_clip * hbr)**2
    is_rem1 = np.logical_and(L1 < l_rem, finite_hbr)
    L1[is_rem1] = l_rem[is_rem1]
    is_rem2 = np.logical_and(L2 < l_rem, finite_hbr)
    L2[is_rem2] = l_rem[is_rem2]
    # L2 is guaranteed to be the smaller of the two eigenvalues, if the
    # remediated value is greater than 0, then the matrix is positive
    # definite
    out["IsPosDef"] = L2 > 0
    out["IsRemediated"] = np.logical_or(is_rem1, is_rem2)
    
    # Sigma values
    sx = np.sqrt(L1)
    sz = np.sqrt(L2)
    out["sx"] = sx
    out["sz"] = sz
    
    # The miss distance coordinates in the conjunction plane (xm,zm)
    # calculated such that both are nonnegative
    rm = np.sqrt(r[:,0]**2 + r[:,1]**2 + r[:,2]**2).reshape(-1, 1)
    xm = rm * np.abs(V1[:,0]).reshape(-1, 1)
    zm = rm * np.abs(V1[:,1]).reshape(-1, 1)
    out["xm"] = xm
    out["zm"] = zm
    
    # Estimate the Pc
    if estimation_mode <= 0:
        # TODO: Need to add unit tests for each of these modes
        if estimation_mode == 0:
            # Mode 0: Calculate the equal-area square Pc approximation,
            # representing the integral over the square with area equal
            # to the HBR circle
            hsq = np.sqrt(np.pi/4.0) * hbr
        else:
            # Mode -1: Calculate the circumscribing square Pc upper
            # bound, representing the integral over the square with area
            # that is always larger and completely encloses the HBR
            # circle
            hsq = hbr
        
        # Calculate the analytical solution for the ensquared Pc, which
        # has an analytical solution involving error functions (Alfano
        # 2005)
        sqrt2 = np.sqrt(2.0)
        dx = sqrt2 * sx
        dz = sqrt2 * sz
        Ex = am.erf_vec_dif( (xm+hsq)/dx, (xm-hsq)/dx)
        Ez = am.erf_vec_dif( (zm+hsq)/dz, (zm-hsq)/dz)
        Pc = Ex * Ez / 4.0
        print("Pc = " + str(Pc))
        
    # TODO: Need to implement estimation modes > 0

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
    params = dict()
    #params["PriSecCovProcessing"] = True
    params["EstimationMode"] = -1
    pc_circle(r1,v1,cov1,r2,v2,cov2,HBR,params)