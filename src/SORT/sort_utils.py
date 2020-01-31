import os
import cv2
import numpy as np
from PIL import Image


def mkdir(path):
    path.strip()
    path.rstrip('\\')
    os.makedirs(path, exist_ok=True)


def save_to_file(root_dic, tracker, frame):
    image_size = 160
    filtered_attrs = [item for item in tracker.face_additional_attribute
                      if item[2] < 1.4 and item[4] < 1]  # recommended threshold values

    out_path = os.path.join(root_dic, str(tracker.id))
    mkdir(out_path)
    for i, item in enumerate(filtered_attrs):
        if item[1] > 0.99:  # face score from MTCNN ,max = 1
            scaled = np.array(Image.fromarray(item[0]).resize((image_size, image_size), resample=Image.BILINEAR))
            scaled = cv2.resize(scaled, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite("%s/%s.jpg" % (out_path, frame - i), scaled)
