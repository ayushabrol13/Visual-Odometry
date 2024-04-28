import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from IPython.display import display, Image
import time
import imageio
import warnings
warnings.filterwarnings('ignore')
import pickle
import io

def SIFT_detector(img1, mask):
    gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

    sift = cv2.SIFT_create() 
    kp, des = sift.detectAndCompute(gray_image, mask) 
     
    return kp, des

def SURF_detector(img, mask):
    det = cv2.SURF_create()
    kp, des = det.detectAndCompute(img, mask)
    return kp, des

def ORB_detector(img, mask):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    orb = cv2.ORB_create(nfeatures=200) 
    kp, des = orb.detectAndCompute(gray_image, mask) 

    return kp, des

def feature_detector(image, detector_type='sift',mask=None):
    if detector_type == 'sift':
        kp, des = SIFT_detector(image, mask)
    if detector_type == 'orb':
        kp, des = ORB_detector(image, mask)
    if detector_type == 'surf':
        kp, des = SURF_detector(image, mask)
    return kp, des

def plot(trajectory, ground_truth):       
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        ax.plot(trajectory[:, 0, 3], 
        trajectory[:, 2, 3], label='estimated', color='green')
        ax.plot(ground_truth[:,0,3], ground_truth[:,2,3])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend

        plt.show()

def calculate_error(estimated, ground_truth):
    estimated = np.array(estimated)
    nframes_est = estimated.shape[0]-1
    se = np.sqrt((ground_truth[nframes_est, 0, 3] - estimated[:, 0, 3])**2 
                    + (ground_truth[nframes_est, 1, 3] - estimated[:, 1, 3])**2 
                    + (ground_truth[nframes_est, 2, 3] - estimated[:, 2, 3])**2)**2
    mse = se.mean()
    return mse

def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t

def compute_depth_map(img_left, img_right, k_left, t_left, t_right, matcher_name='sgbm'):
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(
            numDisparities=num_disparities,
            minDisparity=0,
            blockSize=block_size,
            P1=8 * 3 * sad_window ** 2,
            P2=32 * 3 * sad_window ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16
    
    f = k_left[0][0]

    b = t_right[0] - t_left[0]
    
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left
    
    return depth_map

def stereo_motion_est(match, kp1, kp2, k, depth1, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in match])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]
    object_points = np.zeros((0, 3))
    delete = []

    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(v), int(u)]
        
        if z > max_depth:
            delete.append(i)
            continue
            
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, image1_points, image2_points

def match_features(des1, des2, sort=True, k=2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=k)

    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)

    return matches

def filter_matches(matches, dist_threshold):
    filtered_matches = []
    for m,n in matches:
        ratio = m.distance/n.distance
        if ratio <= dist_threshold:
            filtered_matches.append(m)
    return filtered_matches

def convert_to_homogeneous_matrix(rmat,tvec):
    T_mat = np.eye(4,4)
    T_mat[:3,:3] = rmat
    T_mat[:3,3] = tvec.T
    return T_mat

def stereo_vo_pipeline(left_img_files, right_img_files, P0, P1, matcher='FLANN'):
    num_frames = len(left_img_files)
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    SIFT_det = cv2.SIFT_create()

    for i in range(num_frames-1):
        print(f"Match Number {i}")
        img_left_path = os.path.join('kitti/00/image_0', left_img_files[i]) 
        img_left = cv2.imread(img_left_path)
        img_right_path = os.path.join('kitti/00/image_1', right_img_files[i])
        img_right = cv2.imread(img_right_path)
        next_img_left = cv2.imread(os.path.join('kitti/00/image_0', left_img_files[i+1]))
        
        depth = compute_depth_map(img_left, img_right, k_left, t_left, t_right)
        kp1, des1 = SIFT_det.detectAndCompute(img_left, None)
        kp2,des2 = SIFT_det.detectAndCompute(next_img_left, None)

        matches= match_features(des1, des2)
        filtered_matches = filter_matches(matches, dist_threshold=0.5 )

        R, t, img1_points, img2_points = stereo_motion_est(filtered_matches, kp1, kp2, k_left, depth1=depth)
        
        T_mat = convert_to_homogeneous_matrix(R, t)
        T_tot = T_tot.dot(np.linalg.inv(T_mat))

        trajectory[i+1, :, :] = T_tot[:3, :]
    return trajectory

def save_trajectory_gif(trajectory, ground_truth):
    gif_images = []
    for i in range(len(trajectory)):
        # Plot trajectory
        img = cv2.imread('kitti/00/image_0/' + str(i).zfill(6) + '.png')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = trajectory[:i, 0, 3]
        z = trajectory[:i, 2, 3]
        plt.plot(x, z, color='red', marker='o', linestyle='-', label='trajectory')
        x1 = ground_truth[:i, 0, 3]
        z1 = ground_truth[:i, 2, 3]
        plt.plot(x1, z1, color='blue', marker='o', linestyle='-', label='ground truth')

        # Title and labels
        plt.title(f'Trajectory for frame {i}')

        # Save current frame as image buffer
        plt.savefig('outputs/stereo/temp.png')
        plt.close()
        # Append frame to list  
        gif_images.append(imageio.imread('outputs/stereo/temp.png'))

    # Save frames as GIF
    imageio.mimsave('outputs/stereo/trajectory.gif', gif_images)