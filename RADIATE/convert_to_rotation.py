import numpy as np
from scipy.spatial.transform import Rotation
from geonav_conversion import *

_r = 6378137 # earth radius in meter

def get_scale(lat_deg_0):
    """Get scale s given the first frame's latitude."""
    return np.cos( lat_deg_0*np.pi/180 )

def get_x_y(lat_deg, lon_deg, s):
    """Convert longitude and latitude to x, y euclid coordinate w.r.t. scale using Mercator projection."""
    x = s*_r*np.log( abs(np.tan( np.pi*(90+lat_deg) )) )
    y = s*_r*(np.pi*lon_deg)

    return x, y

def convert_quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to R using scipy."""
    _R = Rotation.from_quat( quaternion.reshape(4, ))
    return np.array(_R.as_matrix())

def convert_IMU_GPU_to_KITTI_pose( GPS_coor_deg, quaternion, origin_lat, origin_lon ):
    """Convert information from IMU and GPS to KITTI format pose.
    
    Args:
        GPS_coor_deg: Latitude, Longitude in degrees, and Altitude.
        quaternion: 4x1 quaternion.
        s: scale obtained from get_scale.

    Returns:
        pose: 4x4: [ R | t ]
                   [ 0 | 1 ]. R: 3x3, t: 3x1    
    """
    _t = np.zeros((3, 1))
    lat, lon, alt = GPS_coor_deg
    
    # _t = lat, lon, alt
    _t = *ll2xy(lat, lon, origin_lat, origin_lon), alt
    
    _R = convert_quaternion_to_rotation_matrix(quaternion)
    pose = np.vstack( [ np.hstack([_R, _t]), np.array([0, 0, 0, 1]).reshape((1, 4)) ] )
    return pose