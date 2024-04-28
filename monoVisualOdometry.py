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


class MonoVisualOdometry():
    def __init__(self, data_dir,K, P):
        self.K= K
        self.P = P
        self.data_dir = data_dir
        # self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = os.listdir(data_dir)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.trajectory_points = []
        self.detected_features = []

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
   
        img1_path = os.path.join(self.data_dir, self.images[i-1])
        img1 = cv2.imread(img1_path,0)
        img2_path = os.path.join(self.data_dir, self.images[i])
        img2 = cv2.imread(img2_path,0)
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(img2, kp1, img1,kp2, good ,None,**draw_params)
        # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        # save_path = os.path.join("outputs/matches_image_0", f"matches_{i}.png")
        # plt.savefig(save_path)
        print(f"Match Number{i}")

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        self.detected_features.append((img2, q1, q2))
        return q1, q2

    def get_pose(self, q1, q2):
   
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
      
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)


            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

      
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t) 

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]
    
    def process(self):
        gt_path = []
        kitti_path = []
        for i in range(len(self.images)):
            if i == 0:
                cur_pose = np.eye(4)
            else:
                q1, q2 = self.get_matches(i)
                transf = self.get_pose(q1, q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
                self.trajectory_points.append((cur_pose[0, 3], cur_pose[2, 3]))

            gt_path.append(cur_pose[:3, :])

        return gt_path, self.trajectory_points

    def save_trajectory_gif(self):
        gif_images = []
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('outputs/MonoVisual', exist_ok=True)
        for i, img_name in enumerate(self.images):
            img = cv2.imread(os.path.join(self.data_dir, img_name))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.plot(*zip(*self.trajectory_points[:i]), marker='o', color='r', linestyle='-')
            plt.axis('off')
            plt.title(f'Trajectory with {i} frames')
            plt.savefig('outputs/MonoVisual/temp.png')
            plt.close()
            gif_images.append(imageio.imread('outputs/MonoVisual/temp.png'))
        imageio.mimsave('outputs/MonoVisual/trajectory.gif', gif_images)

def plot_complete(poses_file_path, trajectory_points, ground_truth):
    gif_frames = []
    for i in range(len(ground_truth)):
        x = ground_truth[:i, 0, 3]
        z = ground_truth[:i, 2, 3]
        img = cv2.imread('kitti/00/image_0/' + str(i).zfill(6) + '.png')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(x, z, color='r', marker = '*',linestyle = '-', label='Ground Truth')
        plt.plot(*zip(*trajectory_points[:i]), color='b', marker ='o', linestyle='-', label='Estimated Trajectory')

        plt.title(f'Trajectory with {i} frames')
        # plt.axis('off')

        plt.savefig('outputs/MonoVisual/full_plot.png')
        plt.close()
        gif_frames.append(imageio.imread('outputs/MonoVisual/full_plot.png'))
    imageio.mimsave('outputs/MonoVisual/full_plot.gif', gif_frames)

def main():
    poses = pd.read_csv('kitti/poses/00.txt', delimiter=' ', header=None)
    ground_truth = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

    left_img_files = os.listdir('kitti/00/image_0') 

    with open('kitti/00/calib.txt', 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('P0:'):
            values = line.split(':')[1].strip().split()
            P = np.array([float(value) for value in values]).reshape(3, 4)
            break

    k_int = np.array([[718.856, 0.   , 607.1928],
                  [0.   , 718.856, 185.2157],
                  [0.    , 0.     , 1.    ]])

    P = np.array([[718.85, 0.0, 607.19, 0], 
                [0.0, 718.85, 185.21, 0],
                [0.0, 0.0, 1.0, 0]])
    
    kitti_data_dir = 'kitti'
    img_files = os.listdir('kitti/00/image_0')

    folder_path = "kitti/00/image_0"

    image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))])
    num_images = min(100, len(image_files))

    images = []
    for i in range(num_images):
        image = imageio.imread(image_files[i])
        images.append(image)
    gif_path = "outputs/MonoVisual/kitti_output.gif"

    kitti_data_dir = 'kitti/00/image_0'
    kitti_vo = MonoVisualOdometry(kitti_data_dir, k_int, P)
    gt_path, trajectory_points = kitti_vo.process()

    kitti_vo.save_trajectory_gif()

    with open('outputs/MonoVisual/gt_path.pkl', 'wb') as f:
        pickle.dump(gt_path, f)

    with open('outputs/MonoVisual/trajctory_points.pkl', 'wb') as f:
        pickle.dump(trajectory_points, f)

    plot_complete('kitti/poses/00.txt', trajectory_points, ground_truth)

    plot(np.array(gt_path), ground_truth)

    print("Error for MonoVisualOdometry:", calculate_error(gt_path, ground_truth))

if __name__ == '__main__':
    main()