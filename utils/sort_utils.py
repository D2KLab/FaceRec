import os
import uuid
from operator import itemgetter

import cv2
import numpy as np
from PIL import Image


def mkdir(path):
    path.strip()
    path.rstrip('\\')
    os.makedirs(path, exist_ok=True)


def save_to_file(root_dic, tracker):
    filter_face_additional_attribute_list = []
    image_size = 160
    for item in tracker.face_addtional_attribute:
        if item[2] < 1.4 and item[4] < 1:  # recommended threshold value
            filter_face_additional_attribute_list.append(item)

    if len(filter_face_additional_attribute_list) > 0:
        score_reverse_sorted_list = sorted(filter_face_additional_attribute_list, key=itemgetter(4))
        for i, item in enumerate(score_reverse_sorted_list):
            if item[1] > 0.99:  # face score from MTCNN ,max = 1
                out_path = os.path.join(root_dic, str(tracker.id))
                mkdir(out_path)

                scaled = np.array(Image.fromarray(item[0]).resize((image_size, image_size), resample=Image.BILINEAR))
                scaled = cv2.resize(scaled, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite("{0}/{1}.jpg".format(out_path, str(uuid.uuid1())), scaled)
