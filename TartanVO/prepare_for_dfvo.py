import os
import argparse
import shutil
from transform_format import transform
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument("--seq", default="P000")
    parser.add_argument("--level", default="Easy")
    parser.add_argument("--remove", default="_left")
    parser.add_argument("--tartanair_root", default="/data/datasets/ntsai/vo/tartanair/")
    parser.add_argument("--scene", default="soulcity")
    parser.add_argument("--calib", default="/data/datasets/ntsai/vo/tartanair/calib.txt")
    parser.add_argument("--dfvo_root", default="/home/ntsai/repos/repo/CMU_16-833_SLAM_Final/DF-VO/dataset/kitti_odom")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    image_dir_trans = {
        "image_left": "image_2",
        "image_right": "image_3"
    }

    in_dir = os.path.join(args.tartanair_root, args.scene, args.scene, args.level, args.seq)

    # Copy calib.txt file
    shutil.copyfile(args.calib, os.path.join(in_dir, "calib.txt"))
    image_dir_name = "image" + args.remove

    original_image_dir = os.path.join(in_dir, image_dir_name)

    # Rename files (remove _left)
    for filename in os.listdir(original_image_dir):
        src = os.path.join(original_image_dir, filename)
        filename_removed = filename.replace(args.remove, "")
        dst = os.path.join(original_image_dir, filename_removed)
        os.rename(src, dst)

    # Rename directory (image_left -> image_2)
    os.rename(original_image_dir, os.path.join(in_dir, image_dir_trans[image_dir_name]))

    # Add to gt to df-vo data
    data_name = f"{args.scene}_{args.seq}"
    gt_path = os.path.join(in_dir, f"pose{args.remove}.txt")
    tartanair_gt = np.loadtxt(gt_path)
    kitti_gt = transform(tartanair_gt, "tartan", "kitti")
    np.savetxt(os.path.join(args.dfvo_root, "gt_poses", f"{data_name}.txt"), kitti_gt)
    
    # Make soft link to df-vo data
    os.symlink(in_dir, os.path.join(args.dfvo_root, "odom_data", data_name))
    

    
    