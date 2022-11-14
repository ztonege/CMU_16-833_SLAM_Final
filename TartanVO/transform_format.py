from Datasets.transformation import kitti2tartan, tartan2kitti
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument("--in_mode", choices=["tartan", "kitti", "df"], default="tartan")
    parser.add_argument("--out_mode", choices=["tartan", "kitti", "df"], default="tartan")
    parser.add_argument("--predicted", default="/home/ntsai/repos/DF-VO/result/kitti_10_rain/10_rain.txt")
    parser.add_argument("--output", default="results/scaled/10_rain_dfvo.txt")
    args = parser.parse_args()

    return args

def transform(est_poses, in_mode, out_mode):
    if in_mode == "tartan":
        est_poses = tartan2kitti(est_poses)
    elif in_mode == "df":
        est_poses = est_poses[:, 1:]

    if out_mode == "tartan":
        est_poses = kitti2tartan(est_poses)
    elif out_mode == "df":
        est_poses = np.hstack(
            [
                np.arange(len(est_poses))[:, None],
                est_poses
            ]
        )
    return est_poses

if __name__ == '__main__':
    args = get_args()

    est_poses = np.loadtxt(args.predicted)
    est_poses = transform(est_poses, args.in_mode, args.out_mode)
    np.savetxt(args.output, est_poses)