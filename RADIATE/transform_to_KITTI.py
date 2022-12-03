from convert_to_rotation import *
import os
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import cv2 as cv
import argparse

def parse_GPS_IMU_Twist(path):
    """Parse a txt file from GPS_IMU_Twist folder and return GPS and Quaternion.
    
    Args:
        path: path to txt file in GPS_IMU_Twist.
    
    Returns:
        GPS: Line 0: Latitude, Longitude, Altitude (in degrees). 3x1 np.ndarray.
        Quaternion: Line 4: IMU orientation. 4x1 np.ndarray.
    """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} not exitst.")

    with open(path, "r") as f:
        data = f.read()    
        data_into_list = data.split("\n")
    
    GPS = np.array( data_into_list[0].split(","), dtype=np.float64).reshape((3, 1))
    Quaternion = np.array( data_into_list[4].split(","), dtype=np.float64).reshape((4, 1))

    return GPS, Quaternion

def get_all_pose_from_dir(dir_path):
    """
    Args:
        dir_path: path to GPS_IMU_Twist folder.

    Returns:
        all_poses: all poses in folder: path. Shape: Nx4x4.
    """
    check_path(dir_path)

    first_frame_path = os.path.join(dir_path, sorted(os.listdir(dir_path))[0])
    _GPS, _ = parse_GPS_IMU_Twist(first_frame_path)
    origin_lat, origin_lon, _ = _GPS

    pose = np.empty((0, 4, 4)) # place holder.
    for file_name in sorted(os.listdir(dir_path)):
        file_path = os.path.join( dir_path, file_name )
        _pose = convert_IMU_GPU_to_KITTI_pose( *parse_GPS_IMU_Twist(file_path), origin_lat, origin_lon )
        pose = np.vstack([pose, _pose[None, :]])

    # t = pose[:, :2, -1].T
    # plt.plot( t[1], t[3] )
    # plt.show()

    return pose

def check_path(path):
    """Check path valid."""
    if not (os.path.isdir(path) or os.path.exists(path))  : raise ValueError(f"Path {path} not exist")

def get_time_stamp(path):
    """Read last column from txt file and return time stamp."""
    with open(path, "r") as f:
        data = f.read()    
        data_into_list = data.split("\n")
    time_stamp = np.array([ _data.split(" ")[-1] for _data in data_into_list ][:-1] ).astype(np.float64)
    return time_stamp

def match_pose_with_image( pose, pose_time_path, image_time_path ):
    """Assign pose to each image."""
    check_path(pose_time_path)
    check_path(image_time_path)

    pose_time_stamp = get_time_stamp(pose_time_path)
    image_time_stamp = get_time_stamp(image_time_path)

    indice = np.zeros_like(image_time_stamp).astype(int)
    for i, _t in enumerate(image_time_stamp):
        idx = np.where( np.abs( _t-pose_time_stamp ) == np.min( np.abs( _t-pose_time_stamp ) )  )
        indice[i] = int(idx[0][0])

    pose = pose[ indice ]
    assert len(pose) == len(image_time_stamp)

    return pose


# def get_stereo_camera_parameters(path):
#     """Load camera paramters from yaml path."""
#     with open(path) as f:
#         calib = yaml.load(f, Loader=SafeLoader)

#     def get_single_camera_parameters(_c):
#         """Retutn camera parametes given calibration parameter dict."""
#         R_vec = np.array(_c["R"])
#         R = cv.Rodrigues(R_vec)[0]
#         T = np.array(_c["T"]).reshape((3, 1))
#         extrinsic = np.hstack( [R, T])
#         K = np.array([[ _c["fx"], 0,       _c["cx"] ],
#                       [ 0,       _c["fy"], _c["cy"] ],
#                       [ 0,        0,       1        ] ])
#         return K@extrinsic

#     P_left  =    get_single_camera_parameters( calib["left_cam_calib"])
#     P_right = get_single_camera_parameters(calib["right_cam_calib"])

#     return P_left, P_right
def get_stereo_camera_parameters(path):
    """Load camera paramters from yaml path."""
    with open(path) as f:
        calib = yaml.load(f, Loader=SafeLoader)

    def get_single_camera_parameters(_c):
        """Retutn camera parametes given calibration parameter dict."""
        K = np.array([[ _c["fx"], 0,       _c["cx"], 0 ],
                      [ 0,       _c["fy"], _c["cy"] , 0],
                      [ 0,        0,       1, 0        ] ])
        return K.reshape(-1, 12)

    P_left  =    get_single_camera_parameters( calib["left_cam_calib"])
    P_right = get_single_camera_parameters(calib["right_cam_calib"])

    return P_left, P_right

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument("--in_dir", default="")
    parser.add_argument("--dfvo_root", default="/home/ntsai/repos/repo/CMU_16-833_SLAM_Final/DF-VO/dataset/kitti_odom")
    parser.add_argument("--seq", default="")
    parser.add_argument("--calib", default="./radiate_sdk/config/default-calib.yaml")
    args = parser.parse_args()

    return args  


if __name__ == '__main__':
    args = get_args()

    seq_dir = os.path.join(args.in_dir, args.seq)
    gps_dir = os.path.join(seq_dir, "GPS_IMU_Twist")
    out_path = os.path.join(seq_dir, f"{args.seq}.txt")
    gps_time_path = os.path.join(seq_dir, "GPS_IMU_Twist.txt")
    zed_time_path = os.path.join(seq_dir, "zed_left.txt")
    calib_path = os.path.join(seq_dir, "calib.txt")

    # Get pose and match points
    pose = get_all_pose_from_dir(gps_dir)
    matched_points = match_pose_with_image(pose, gps_time_path, zed_time_path)
    p_left, p_right = get_stereo_camera_parameters(args.calib)

    matched_points = matched_points[:, [1, 2, 0]]
    matched_points = matched_points.reshape(-1, 12)

    # Create calib file
    calibs = [
        list(p_left.flatten().astype("str")), 
        list(p_right.flatten().astype("str")), 
        list(p_left.flatten().astype("str")), 
        list(p_right.flatten().astype("str"))
    ]
    for i, calib in enumerate(calibs):
        calibs[i] = [f"P{i}"] + calibs[i]
        calibs[i] = " ".join(calibs[i]) + "\n"

    # Save poses and calib
    np.savetxt(out_path, matched_points)
    with open(calib_path, "w+") as file:
        file.writelines(calibs)

    # Rename to image_2
    image_dir_old = os.path.join(seq_dir,  "zed_left")
    image_dir_new = os.path.join(seq_dir, "image_2")
    if os.path.exists(image_dir_old):
        os.rename(image_dir_old, image_dir_new)
    
    # Start image with 000000
    if not os.path.exists(os.path.join(image_dir_new, "000000.png")):
        for i in range(len(matched_points)):
            image_path_old = os.path.join(image_dir_new, str(i+1).zfill(6) + ".png")
            image_path_new = os.path.join(image_dir_new, str(i).zfill(6) + ".png")
            os.rename(image_path_old, image_path_new)

    # Create soft link to gt
    # Make soft link to df-vo data
    os.symlink(seq_dir, os.path.join(args.dfvo_root, "odom_data", args.seq))
    # Create soft link to data
    os.symlink(out_path, os.path.join(args.dfvo_root, "gt_poses", f"{args.seq}.txt"))

    

# def read_time(path):
#     stamps = []
#     with open(path, "r") as file:
#         for line in file.readlines():
#             stamp = line.split()[3]
#             if len(stamp) < 20:
#                 int_part, dec_part = stamp.split(".")
#                 dec_part = dec_part.zfill(9)
#                 stamp = ".".join([int_part, dec_part])
#             stamps.append(float(stamp))
#     return np.array(stamps)

# def match_times(gps_time_path, zed_time_path):
#     gps_time = read_time(gps_time_path)
#     zed_time = read_time(zed_time_path)
#     zed_idx, gps_idx = 0, 0
#     prev_diff = float("inf")
#     match_idx = []
#     while zed_idx < len(zed_time) and gps_idx < len(gps_time):
#         cur_diff = abs(zed_time[zed_idx] - gps_time[gps_idx])
#         if cur_diff <= prev_diff:
#             prev_diff = cur_diff
#             gps_idx += 1
#         else:
#             print(cur_diff, prev_diff)
#             match_idx.append(gps_idx)
#             prev_diff = float("inf")
#             zed_idx += 1
#             gps_idx += 1
#     print(zed_idx, gps_idx)
#     print(np.hstack([zed_time[:, None], gps_time[match_idx][:, None]]))
#     np.savetxt("tmp.txt", np.hstack([zed_time[:, None], gps_time[match_idx][:, None]]))
#     raise
#     return match_idx