"""Provides probability of collision calculations
"""

# import numpy as np
# from cara_analysis_tools.utils.datatypes import (
#     MatrixType,
#     VectorType,
#     valid_cov_matrix,
#     valid_vector,
#     )

# def pc_circle(r1: VectorType, v1: VectorType, C1: MatrixType,
#               r2: VectorType, v2: VectorType, C2: MatrixType,
#               hbr: float, params: dict = {}) -> tuple[float, dict]:
#     """Computes Pc for state/cov input by integrating over a circle on
#     the conjunction plane.
    
#     TODO: Need to fill in with details once implemented

#     Parameters
#     ----------
#     r1 : VectorType
#         Primary object's position vector in inertial cartesian
#         coordinates, size is 1x3.
#     v1 : VectorType
#         Primary object's velocity vector in inertial cartesian
#         coordinates, size is 1x3.
#     C1 : MatrixType
#         Primary object's covariance matrix in the same inertial
#         cartesian coordinate frame as the position and velocity, size is
#         3x3, 6x6, or nxn with n > 6.
#     r2 : VectorType
#         Secondary object's position vector in inertial cartesian
#         coordinates, size is 1x3.
#     v2 : VectorType
#         Secondary object's velocity vector in inertial cartesian
#         coordinates, size is 1x3.
#     C2 : MatrixType
#         Secondary object's covariance matrix in the same inertial
#         cartesian coordinate frame as the position and velocity, size is
#         3x3, 6x6, or nxn with n > 6.
#     hbr : float
#         Coimbined hard body radius of the primary and secondary objects.
#     params : dict, optional
#         TODO Fill this section out if needed
#         _description_, by default {}

#     Returns
#     -------
#     tuple[float, dict]
#         Pc - Computed probability of collision
#         out - Dictionary containing the following supplemental
#               information:
#           IsPosDef - Flag indicating if the combined and marginalized
#                      covariance has a negative eigenvalue.
#           IsRemediated - Flag indicating if the combined and
#                          marginalized 2x2 covariance was remediated,
#                          either successfully or not.
#           Amat - Combined covariance projected onto the nominal
#                  conjunction plane.
#           xm, zm - Position of the mean relative miss distance on the
#                    conjunction plane.
#           sx, sz - Sigma values of the relative miss distance PDF on the
#                    conjunction plane.
#           r1,v1,C1,r2,v2,C2,hbr - Input parameters saved off for use in
#                                   other functions.
    
#     Raises
#     ------
#     ValueError
#         Occurs when invalid covariance matrix or vectors are passed in.
    
#     References
#     ----------
#     Alfano, S. "A Numerical Implementation of Spherical Object Collision
#     Probability." Journal of the Astronautical Sciences, Vol. 53, No. 1,
#     pp. 103-109, Jan-Mar 2005.
#     """
    
#     # Check for valid vectors and matrices
#     if not valid_vector(r1):
#         raise ValueError("r1 vector must be 1x3 NDarray")
#     if not valid_vector(v1):
#         raise ValueError("v1 vector must be 1x3 NDarray")
#     if not valid_cov_matrix(C1):
#         raise ValueError("C1 matrix must be 3x3, 6x6, or nxn (n>6)"
#                          + " symmetric NDarray")
#     if not valid_vector(r2):
#         raise ValueError("r2 vector must be 1x3 NDarray")
#     if not valid_vector(v2):
#         raise ValueError("v2 vector must be 1x3 NDarray")
#     if not valid_cov_matrix(C2):
#         raise ValueError("C2 matrix must be 3x3, 6x6, or nxn (n>6)"
#                          + " symmetric NDarray")
#     c1_size = np.size(C1)
#     c2_size = np.size(C2)
#     if c1_size[0] != c2_size[0]:
#         raise ValueError("C1 and C2 matrices must be the same size")
    
#     # Save the input parameters into the output structure
#     out = {}
#     out["r1"] = r1
#     out["v1"] = v1
#     out["C1"] = C1
#     out["r2"] = r2
#     out["v2"] = v2
#     out["C2"] = C2
#     out["hbr"] = hbr
    
#     # Combine the covariances
#     comb_cov = C1 + C2
    
#     # Relative position and velocity
#     r = r1 - r2
#     v = v1 - v2
    
#     # Check and adjust for zero miss distance (for processing Alfano
#     # 2009 test cases)
#     rmag = np.linalg.norm(r)
#     reps = max(10 * np.spacing(rmag), 1.0e-6*hbr)
#     if rmag < reps:
#         rsum = r1 + r2
#         rsum_mag = np.linalg.norm(rsum)
#         vmag = np.linalg.norm(v)
#         rdel = reps * np.cross(rsum,v) / rsum_mag / vmag
#         r = r + rdel
    
#     # Check for zero relative velocity (for processing Alfano 2009 test
#     # cases)
#     vmag = np.linalg.norm(v)
#     if vmag == 0:
#         # TODO display warning for zero rel vel and set Pc to NaN
#         dummy = 1
    
#     # Orbit normal
#     h = np.cross(r,v)
    
#     # Construct the relative encounter frame
#     y = v / np.linalg.norm(v)
#     z = h / np.linalg.norm(h)
#     x = np.cross(y, z)
#     eci2xyz = np.stack((x, y, z), axis = 1)
#     out["xhat"] = x
#     out["yhat"] = y
#     out["zhat"] = z
