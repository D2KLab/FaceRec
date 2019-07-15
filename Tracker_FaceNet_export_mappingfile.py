import argparse
import os

import align.detect_face as detect_face
import cv2
import numpy as np
import tensorflow as tf
from utils.face_utils import judge_side_face
from utils.sort_utils import mkdir
from SORT.sort import Sort
from statistics import mode
import utils.facenet as facenet
import pickle
import math
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier

def main():
    global colours, img_size
    args = parse_args()
    video_dir = args.video_dir
    output_path = args.output_path
    all_trackers_saved = args.all_trackers_saved
    obid_mapping_classnames = args.obid_mapping_classnames
    #classifier_path = "classifier\\knn_classifier_n1.pkl"
    classifier_filename = args.classifer_path
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file")
    #with open(classifier_path, 'rb') as f:
        #(model, class_names) = pickle.load(f)
        #print("Loaded classifier file")
    #output_path = "data\\output_label"
    
    # for disp

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker


    print('Start track and extract......')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False)) as sess:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
            facenet_model_path = args.model_path
            #facenet_model_path ="model\\20180402-114759.pb"
            facenet.load_model(facenet_model_path)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            margin = 40 # if the face is big in your video ,you can set it bigger for tracking easiler            
            minsize = 50 # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            input_image_size = 160
            frame_interval = 1  # interval how many frames to make a detection,you need to keep a balance between performance and fluency
            scale_rate = 0.9  # if set it smaller will make input frames smaller
            show_rate = 0.8  # if set it smaller will dispaly smaller frames
           

            for filename in os.listdir(video_dir):
                suffix = filename.split('.')[1]
                if suffix != 'mp4' and suffix != 'avi':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(video_dir, filename)
                directoryname = os.path.join(output_path, filename.split('.')[0])
                
                cam = cv2.VideoCapture(video_name)
                c = 0
                while True:
                    final_faces = []
                    addtional_attribute_list = []
                    ret, frame = cam.read()
                    if not ret:
                        print("ret false")
                        break
                    if frame is None:
                        print("frame drop")
                        break

                    frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if c % frame_interval == 0:
                        img_size = np.asarray(frame.shape)[0:2]
                        faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold, factor)
                        face_sums = faces.shape[0]
                        if face_sums > 0:
                            face_list = []
                            for i, item in enumerate(faces):
                                f = round(faces[i, 4], 6)
                                if f > 0.99:
                                    det = np.squeeze(faces[i, 0:4])
                                    
                                    face_list.append(item)

                                    # face cropped
                                    bb = np.array(det, dtype=np.int32)
                                    frame_copy = frame.copy()
                                    cropped = frame_copy[bb[1]:bb[3], bb[0]:bb[2], :]

                                    # use 5 face landmarks  to judge the face is front or side
                                    squeeze_points = np.squeeze(points[:, i])
                                    tolist = squeeze_points.tolist()
                                    facial_landmarks = []
                                    for j in range(5):
                                        item = [tolist[j], tolist[(j + 5)]]
                                        facial_landmarks.append(item)
                                    
                                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                        np.array(facial_landmarks))

                                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                    item_list = [cropped, faces[i, 4], dist_rate, high_ratio_variance, width_rate]
                                    addtional_attribute_list.append(item_list)

                            final_faces = np.array(face_list)

                    trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, r_g_b_frame)
                    #with open(r'all_tracker_saved_non_negative.txt', 'a+') as f:
                        #f.write(" ".join(map(str, trackers)) + "\n")
                    c += 1
                    for d in trackers:
                        obid_map_classname = []
                        print(d)
                        d = [int(i) for i in d]
                        print(d)
                        if all(i >= 0 for i in d):
                            with open(all_trackers_saved, 'a+') as f:
                                f.write(" ".join(map(str, d)) +'.'+str(c)+ "\n")                                
                            trackers_cropped = frame[d[1]:d[3], d[0]:d[2], :]
                            try:
                                scaled = cv2.resize(trackers_cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                            except Exception as e:
                                print('the broken image')   
                            #scaled = cv2.resize(trackers_cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            if best_class_probabilities > 0.09:
                                with open(obid_mapping_classnames, 'a+') as f:
                                    f.write(best_name+'.'+ str(d[4]) + "\n")                                     
                        else:
                            with open(obid_mapping_classnames, 'a+') as f:
                                f.write('error tracker' +str(c)+ "\n")                      

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str,
                        help='Path to the data directory containing videos.',
                        default = "./video")
    parser.add_argument('--output_path', type=str,
                        help='Path to the cluster folder',
                        default='data/cluster')
    parser.add_argument('--all_trackers_saved', type=str,
                        help='Path to the txt file for all trackers saved')
    parser.add_argument('--obid_mapping_classnames', type=str,
                        help='Path to the txt output file for mapping file')    
    parser.add_argument('--classifer_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier_1NN_grayscale46891.pkl")
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    args = parser.parse_args()
    return args
    
    #parser.add_argument("--videos_dir", type=str,help='Path to the data directory containing aligned your face patches.', default='video')
    #args = parser.parse_args()
    #return args
                                    
if __name__ == '__main__':                                   
    main()                                                           
