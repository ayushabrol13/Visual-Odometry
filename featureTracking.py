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

def normalize_points(points, K_inv):
    # Normalize image points to get normalized image coordinates
    homogeneous_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return np.dot(homogeneous_points, K_inv.T)

def essential_matrix(kp1, kp2, K):
    K_inv = np.linalg.inv(K)
    
    # Normalize keypoints
    norm_kp1 = normalize_points(kp1, K_inv)
    norm_kp2 = normalize_points(kp2, K_inv)
    
    A = np.array([
        [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]
        for (u1, v1), (u2, v2) in zip(norm_kp1, norm_kp2)
    ])
    
    _, _, V = np.linalg.svd(A)
    F_flat = V[-1]
    F = F_flat.reshape(3, 3)
    
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    
    E = np.dot(K.T, np.dot(F, K))
    
    return E

class VisualOdometryTracking():
    def __init__(self,kmin):
        self.kMinNumFeature = kmin
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)
        self.x_loc = []
        self.z_loc = []
        self.cur_R = None
        self.cur_t = None
        self.trajectory = None
        self.trajectory_points = []

    def convert_to_homogeneous_matrix(self, rmat, tvec):
        T_mat = np.eye(4, 4)
        T_mat[:3, :3] = rmat
        T_mat[:3, 3] = tvec.T
        return T_mat

    def featureTracking(self, image_ref, image_cur, px_ref):
        lk_params = dict(winSize=(21, 21),
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        return kp1, kp2

    def process_first_frames(self, first_frame, second_frame, k):
        det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp1 = det.detect(first_frame)
        kp1 = np.array([x.pt for x in kp1], dtype=np.float32)

        kp1, kp2 = self.featureTracking(first_frame, second_frame, kp1)
        E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
        kp1 = kp2
        return kp1, R, t

    def feature_tracking_vo_pipeline(self,data_dir, left_img_files, initial, k):
        first_frame_path = os.path.join(data_dir, left_img_files[initial])
        first_frame = cv2.imread(first_frame_path, 0)
        second_frame_path = os.path.join(data_dir, left_img_files[initial + 1])
        second_frame = cv2.imread(second_frame_path, 0)
        kp1, self.cur_R, self.cur_t = self.process_first_frames(first_frame, second_frame, k)
        last_frame = second_frame
        self.trajectory = np.zeros((len(left_img_files), 3, 4))
        T = np.eye(4)
        self.trajectory[0] = T[:3, :]

        for i in range(len(left_img_files) - 1):
            print(f"Match Number {i}")
            new_img_path = os.path.join(data_dir, left_img_files[i])
            new_frame = cv2.imread(new_img_path, 0)
            kp1, kp2 = self.featureTracking(last_frame, new_frame, kp1)
            E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)

            change = np.mean(np.abs(kp2 - kp1))

            if change > 5:
                self.cur_t = self.cur_t + 1 * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
            if kp1.shape[0] < self.kMinNumFeature:
                det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
                kp2 = det.detect(new_frame)
                kp2 = np.array([x.pt for x in kp2], dtype=np.float32)

            kp1 = kp2

            last_frame = new_frame
            if i > 2:
                tvec = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]]).reshape(3, 1)
            else:
                tvec = np.array([0, 0, 0]).reshape(3, 1)

            T_mat = self.convert_to_homogeneous_matrix(self.cur_R, self.cur_t)

            self.trajectory[i + 1, :, :] = T_mat[:3, :]

        return self.trajectory
    
    def save_trajectory_gif(self, ground_truth):
        gif_images = []
        for i in range(len(self.trajectory)):
            # Plot trajectory
            img = cv2.imread('kitti/00/image_0/' + str(i).zfill(6) + '.png')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            x = self.trajectory[:i, 0, 3]
            z = self.trajectory[:i, 2, 3]
            plt.plot(x, z, color='red', marker='o', linestyle='-', label='trajectory')
            x1 = ground_truth[:i, 0, 3]
            z1 = ground_truth[:i, 2, 3]
            plt.plot(x1, z1, color='blue', marker='o', linestyle='-', label='ground truth')

            # Title and labels
            plt.title(f'Trajectory for frame {i}')

            # Save current frame as image buffer
            plt.savefig('outputs/feature_tracking/temp.png')
            plt.close()
            # Append frame to list  
            gif_images.append(imageio.imread('outputs/feature_tracking/temp.png'))

        # Save frames as GIF
        imageio.mimsave('outputs/feature_tracking/trajectory.gif', gif_images)

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/feature_tracking', exist_ok=True)
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
    gif_path = "outputs/feature_tracking/kitti_output.gif"
    
    data_dir = 'kitti/00/image_0'

    vo = VisualOdometryTracking(kmin= 1000)
    left_img_files = os.listdir(data_dir)
    trajectory = vo.feature_tracking_vo_pipeline(data_dir, left_img_files, 0, k_int)
    
    with open('outputs/feature_tracking/trajectory.pkl', 'wb') as f:
        pickle.dump(trajectory, f)

    vo.save_trajectory_gif(ground_truth)

    plot(trajectory, ground_truth)

    print("Error for VisualOdometryTracking:", calculate_error(trajectory, ground_truth))

if __name__ == '__main__':
    main()