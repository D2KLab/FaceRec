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

from .align import detect_face
from .utils import facenet, utils
from .SORT.sort import Sort
from .utils.face_utils import judge_side_face

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")


def main(video_path, output_path, classifier_path='classifier/classifier.pkl',
         facenet_model_path='model/20180402-114759.pb', video_speedup=1, prob_threshold=0.7,
         dominant_ratio=0.5, merge_cluster=False):
    video_capture = utils.get_capture(video_path)

    if output_path is None:
        output_path = utils.generate_output_path('./data/cluster', video_path)

    minsize = 50  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    input_image_size = 160
    scale_rate = 0.9  # if set it smaller will make input frames smaller

    # Get the path of the facenet model and load it
    facenet.load_model(facenet_model_path)

    # Load classifier
    classifier_filename = os.path.expanduser(classifier_path)
    with open(classifier_filename, 'rb') as f:
        (classifier, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Bounding box
            pnet, rnet, onet = detect_face.create_mtcnn(sess, ALIGN_MODEL_PATH)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # frames per second
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_passed = -1

            # start reading frame by frame
            while video_capture.grab():  # move pointer to next frame
                total_frames_passed += 1
                # Skip frames if video is to be speed up
                if video_speedup > 1 and total_frames_passed % video_speedup != 0:
                    continue

                # Otherwise read the frame
                ret, frame = video_capture.retrieve()

                face_list = []
                additional_attribute_list = []
                frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img_size = np.asarray(frame.shape)[0:2]
                bounding_boxes, points = detect_face.detect_face(rgb_frame, minsize, pnet, rnet, onet,
                                                                 threshold, factor)
                # points are the face landmarks

                for i, item in enumerate(bounding_boxes):
                    f = round(item[4], 6)
                    if f > 0.99:
                        det = np.squeeze(item[0:4])
                        face_list.append(item)

                        # face cropped
                        bb = np.array(det, dtype=np.int32)
                        cropped = frame.copy()[bb[1]:bb[3], bb[0]:bb[2], :]

                        # use 5 face landmarks to judge the face is front or side
                        plist = np.squeeze(points[:, i]).tolist()
                        facial_landmarks = [[plist[j], plist[(j + 5)]] for j in range(5)]

                        dist_rate, high_ratio_variance, width_rate = judge_side_face(
                            np.array(facial_landmarks))

                        # face additional attribute
                        # (index 0:face score; index 1:0 represents front face and 1 for side face )
                        additional_attribute_list.append(
                            [cropped, item[4], dist_rate, high_ratio_variance, width_rate])

                final_faces = np.array(face_list)
                trackers = tracker.update(final_faces, img_size, output_path, additional_attribute_list, rgb_frame)

            print("Finished making the cluster...")
            with open('tracker_saved_greater_5.txt') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            time_dict = dict()
            for i in content:
                id_cluster = int(i.split('.')[0])
                id_frame = int(i.split('.')[1])
                time_dict.update({id_cluster: id_frame})

            dataset = facenet.get_dataset(output_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            label_name = [cls.name.replace('_', ' ') for cls in dataset]
            labels = [label_name[i] for i in labels]
            print('Number of cluster: {}'.format(len(dataset)))
            print('Number of images: {}'.format(len(paths)))
            print('Loading feature extraction model')

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
            emb_array_added_labels = np.hstack((emb_array, np.atleast_2d(labels).T))
            emb_array_added_labels = pd.DataFrame(data=emb_array_added_labels)
            embedding_training = pd.read_csv('data/embedding/embedding.csv', header=None)
            label_training = pd.read_csv('data/embedding/label.csv', header=None)
            embedding_training_added_labels = pd.concat([embedding_training, label_training], axis=1,
                                                        ignore_index=True)
            embedding_training_added_labels = embedding_training_added_labels.drop(
                embedding_training_added_labels.columns[[-1]], axis=1)

            print('for testing...')
            predictions = classifier.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            index_great = []
            for idx, val in enumerate(best_class_probabilities):
                if val > prob_threshold:
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
                    if value.count(dominant) / len(value) * 1.0 > dominant_ratio:
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
                                                                              output_path)
                print("{}: {}".format(famous_person, cluster_id_famous_person_list))
                if cluster_id_famous_person_list:
                    count_frames_for_cluster(cluster_id_famous_person_list, output_path)
                    for i in cluster_id_famous_person_list:
                        emb_array_training = cluster_subset(embedding_training_added_labels,
                                                            [i for i, x in enumerate(class_names) if
                                                             x == famous_person][0])
                        emb_array_testing = cluster_subset(emb_array_added_labels, i)
                        list1 = distance_among_2_clusters(emb_array_testing, emb_array_training)
                        print("Cluster ID {} - {} - Cosine distance Mean: {}".format(i, famous_person,
                                                                                     mean3maxelements(list1)))
                for i in cluster_id_famous_person_list:
                    print("cluster Id {} - {} appear on {}".format(i, famous_person, str(
                        datetime.timedelta(seconds=time_dict[int(i)] * 1 / fps))))


def check_consecutive(l):
    return sorted(l) == list(range(min(l), max(l) + 1))


def merge_folders(root_src_dir, root_dst_dir):
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
                if check_consecutive([i, j]):
                    src_dir = os.path.join(directoryname, str(j))
                    dst_dir = os.path.join(directoryname, str(i))
                    merge_folders(src_dir, dst_dir)
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
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument('--video_speedup', type=int,
                        help='Speed up for the video', default=1)
    parser.add_argument("--threshold", type=float,
                        help='threshold',
                        default=0.7)
    parser.add_argument('--output_path', type=str,
                        help='Path to the cluster folder',
                        default='data/cluster')
    parser.add_argument('--classifier_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier.pkl")
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--dominant_ratio', type=float,
                        help='Ratio threshold to decide cluster name', default=0.5)
    parser.add_argument('--merge_cluster', default=False, action='store_true',
                        help='Include the argument for merging the clusters')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.video, args.output_path, args.classifier_path, args.model_path, args.video_speedup, args.threshold,
         args.dominant_ratio, args.merge_cluster)
