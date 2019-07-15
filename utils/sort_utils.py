
import os
import uuid
from operator import itemgetter
from scipy import misc

import cv2



def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def save_to_file(root_dic, tracker):
    filter_face_addtional_attribute_list = []
    input_image_size = 160
    image_size = 182
    for item in tracker.face_addtional_attribute:
        if item[2] < 1.4 and item[4] < 1: # recommended thresold value
            filter_face_addtional_attribute_list.append(item)
    if (len(filter_face_addtional_attribute_list) > 0):
        score_reverse_sorted_list = sorted(filter_face_addtional_attribute_list, key=itemgetter(4))
        for i, item in enumerate(score_reverse_sorted_list):
            if item[1] > 0.99: # face score from MTCNN ,max = 1
                out_path = os.path.join(root_dic,str(tracker.id))
                mkdir(out_path)
                try:
                    scaled = misc.imresize(item[0], (input_image_size, input_image_size), interp='bilinear')
                    scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)                
                    
                    cv2.imwrite(
                        "{0}/{1}.jpg".format(out_path, str(uuid.uuid1())), scaled)
                except:
                    print('warning: Invalid image size.')
                


