import numpy as np


def judge_side_face(facial_landmarks):
    fl = np.array(facial_landmarks)
    wide_dist = np.linalg.norm(fl[0] - fl[1])
    high_dist = np.linalg.norm(fl[0] - fl[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = fl[0] - fl[2]
    vec_B = fl[1] - fl[2]
    vec_C = fl[3] - fl[2]
    vec_D = fl[4] - fl[2]
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
