import numpy as np


def judge_side_face(facial_landmarks):
    fl = facial_landmarks
    print('here')
    wide_dist = np.linalg.norm(np.subtract(fl['left_eye'], fl['right_eye']))
    high_dist = np.linalg.norm(np.subtract(fl['left_eye'], fl['mouth_left']))
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = np.subtract(fl['left_eye'], fl['nose'])
    vec_B = np.subtract(fl['right_eye'], fl['nose'])
    vec_C = np.subtract(fl['mouth_left'], fl['nose'])
    vec_D = np.subtract(fl['mouth_right'], fl['nose'])
    dist_A = np.linalg.norm(vec_A)
    dist_B = np.linalg.norm(vec_B)
    dist_C = np.linalg.norm(vec_C)
    dist_D = np.linalg.norm(vec_D)

    # cal rate
    high_rate = dist_A / dist_C
    width_rate = dist_C / dist_D
    high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
    width_ratio_variance = np.fabs(width_rate - 1)

    return dist_rate, high_ratio_variance, width_ratio_variance
