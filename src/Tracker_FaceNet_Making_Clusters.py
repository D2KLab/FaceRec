import argparse
import datetime
import math
import os
import pickle
import shutil
from statistics import mean
from statistics import mode

import cv2
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from scipy.spatial import distance

import align.detect_face as detect_face
import utils.facenet as facenet
from SORT.sort import Sort
from utils.face_utils import judge_side_face


def main():
    global colours, img_size
    args = parse_args()
    video_dir = args.video_dir
    output_path = args.output_path
    merge_cluster = args.merge_cluster
    tracker = Sort()  # create instance of the SORT tracker

    print('Start track and extract......')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
            facenet_model_path = args.model_path
            facenet.load_model(facenet_model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            margin = 40  # if the face is big in your video ,you can set it bigger for tracking easiler
            minsize = 50  # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            input_image_size = 160
            frame_interval = args.frame_interval  # interval how many frames to make a detection
            scale_rate = 0.9  # if set it smaller will make input frames smaller
            show_rate = 0.8  # if set it smaller will dispaly smaller frames

            for filename in os.listdir(video_dir):
                suffix = filename.split('.')[-1]
                if suffix != 'mp4' and suffix != 'avi':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(video_dir, filename)
                directory_name = os.path.join(output_path, filename.split('.')[0])

                cam = cv2.VideoCapture(video_name)
                fps = cam.get(cv2.CAP_PROP_FPS)
                total_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
                c = 0
                # while True:
                for a in [0, 10000]:
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
                        faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                                factor)
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

                                    # use 5 face landmarks to judge the face is front or side
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

                    trackers = tracker.update(final_faces, img_size, directory_name, addtional_attribute_list,
                                              r_g_b_frame)
                    c += 1
                print("Finished making the cluster...")
                with open('tracker_saved_greater_5.txt') as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                time_dict = dict()
                for i in content:
                    id_cluster = int(i.split('.')[0])
                    id_frame = int(i.split('.')[1])
                    time_dict.update({id_cluster: id_frame})

                dataset = facenet.get_dataset(directory_name)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                label_name = [cls.name.replace('_', ' ') for cls in dataset]
                labels = [label_name[i] for i in labels]
                print('Number of cluster: {}'.format(len(dataset)))
                print('Number of images: {}'.format(len(paths)))
                print('Loading feature extraction model')
                embedding_model_dir = args.model_path
                classifier_filename = args.classifier_path
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (classifier, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)
                facenet.load_model(embedding_model_dir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print(embedding_size)
                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                batch_size = 200
                image_size = 160
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    print('{}/{}'.format(i + 1, nrof_batches_per_epoch))
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                emb_array_added_lables = np.hstack((emb_array, np.atleast_2d(labels).T))
                emb_array_added_lables = pd.DataFrame(data=emb_array_added_lables)
                embedding_training = pd.read_csv('data/embedding/embedding.csv', header=None)
                label_training = pd.read_csv('data/embedding/label.csv', header=None)
                embedding_training_added_lables = pd.concat([embedding_training, label_training], axis=1,
                                                            ignore_index=True)
                embedding_training_added_lables = embedding_training_added_lables.drop(
                    embedding_training_added_lables.columns[[-1]], axis=1)

                print('for testing...')
                predictions = classifier.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                index_great = []
                for idx, val in enumerate(best_class_probabilities):
                    if val > args.threshold:
                        index_great.append(idx)
                prediction_new = []
                for idx, val in enumerate(best_class_indices):
                    if idx in index_great:
                        prediction_new.append(val)
                labels_new = []
                for idx, val in enumerate(labels):
                    if idx in index_great:
                        labels_new.append(val)
                cluster_result = {}
                interest_cluster = {}

                for a, b in zip(labels_new, prediction_new):
                    cluster_result.setdefault(a, []).append(b)
                for key, value in cluster_result.items():
                    try:
                        dominant = mode(value)
                        if value.count(dominant) / len(value) * 1.0 > args.dominant_ratio:
                            name = class_names[dominant]
                            interest_cluster.update({key: name})
                            # print("cluster Id {} : {} appear on {}".format(key,name,str(datetime.timedelta(seconds=time_dict[int(key)]*1/fps))))
                    except:
                        name = "Unknow"
                print("Total frames of the video:", total_frame)
                print("The frame rate of the video:", fps)
                famous_persons = []
                for i, j in interest_cluster.items():
                    famous_persons.append(j)
                famous_persons = list(set(famous_persons))
                for famous_person in famous_persons:
                    cluster_id_famous_person_list = []
                    for i, j in interest_cluster.items():
                        if j == famous_person:
                            cluster_id_famous_person_list.append(i)
                    if merge_cluster:
                        cluster_id_famous_person_list = merge_consecutive_cluster(cluster_id_famous_person_list,
                                                                                  directory_name)
                    print("{}: {}".format(famous_person, cluster_id_famous_person_list))
                    if cluster_id_famous_person_list:
                        count_frames_for_cluster(cluster_id_famous_person_list, directory_name)
                        for i in cluster_id_famous_person_list:
                            emb_array_training = cluster_subset(embedding_training_added_lables,
                                                                [i for i, x in enumerate(class_names) if
                                                                 x == famous_person][0])
                            emb_array_testing = cluster_subset(emb_array_added_lables, i)
                            list1 = distance_among_2_clusters(emb_array_testing, emb_array_training)
                            print("Cluster ID {} - {} - Cosine distance Mean: {}".format(i, famous_person,
                                                                                         mean3maxelements(list1)))
                    for i in cluster_id_famous_person_list:
                        print("cluster Id {} - {} appear on {}".format(i, famous_person, str(
                            datetime.timedelta(seconds=time_dict[int(i)] * 1 / fps))))


def checkConsecutive(l):
    return sorted(l) == list(range(min(l), max(l) + 1))


def mergefolders(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
        shutil.rmtree(root_src_dir)


def merge_consecutive_cluster(humanname, directoryname):
    humanname = list(map(int, humanname))
    for i in humanname:
        for j in humanname:
            if i < j:
                if checkConsecutive([i, j]):
                    src_dir = os.path.join(directoryname, str(j))
                    dst_dir = os.path.join(directoryname, str(i))
                    mergefolders(src_dir, dst_dir)
                    humanname.remove(j)
                    break
    humanname = list(map(str, humanname))
    return humanname


def count_frames(path):
    frames = 0
    for _, _, filenames in os.walk(path):
        frames += len(filenames)
        print(frames)


def count_frames_for_cluster(humanname, directoryname):
    if humanname:
        for i in humanname:
            print("The number of frames of cluster {}: ".format(i), end=" ")
            count_frames(os.path.join(directoryname, i))


def compare(humanname, mean_dict):
    if humanname:
        for i in range(len(humanname)):
            for j in range(len(humanname)):
                print("Cosine similarity {} - {}: {}".format(humanname[i], humanname[j],
                                                             distance.cosine(mean_dict[float(humanname[i])],
                                                                             mean_dict[float(humanname[j])])))
                print("Euclidean distance {} - {}: {}".format(humanname[i], humanname[j],
                                                              distance.euclidean(mean_dict[float(humanname[i])],
                                                                                 mean_dict[float(humanname[j])])))


def cluster_subset(emb_array_added_lables, labels_value):
    emb_array_lb = emb_array_added_lables[emb_array_added_lables[512] == labels_value]
    emb_array_lb = emb_array_lb.drop(emb_array_lb.columns[[-1]], axis=1)
    emb_array_lb = emb_array_lb.values
    emb_array_lb = emb_array_lb.astype(np.float)
    return emb_array_lb


def mean3maxelements(list1):
    final_list = []
    for i in range(3):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]
        list1.remove(max1)
        final_list.append(max1)
    return mean(final_list)


def distance_among_2_clusters(a, b):
    distance1 = []
    for i in a:
        for j in b:
            k = 1 - distance.cosine(i, j)
            distance1.append(k)
    return distance1


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str,
                        help='Path to the data directory containing videos.',
                        default="./video")
    parser.add_argument("--frame_interval", type=int,
                        help='interval how many frames to make a detection',
                        default=1)
    parser.add_argument("--threshold", type=float,
                        help='threshold',
                        default=0.7)
    parser.add_argument('--output_path', type=str,
                        help='Path to the cluster folder',
                        default='data/cluster')
    parser.add_argument('--classifier_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier_1NN_grayscale46891.pkl")
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--dominant_ratio', type=float,
                        help='Ratio threshold to decide cluster name', default=0.5)
    parser.add_argument('--merge_cluster', type=int,
                        help='Whether merge cluster or not', default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
