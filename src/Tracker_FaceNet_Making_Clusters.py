import argparse
import math
import os
import pickle
import shutil
from statistics import mean, mode, StatisticsError

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from scipy.spatial import distance

from .utils import facenet, utils

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")


def main(video_path, face_fragment_path, classifier_path='classifier/classifier.pkl',
         facenet_model_path='model/20180402-114759.pb', prob_threshold=0.7,
         dominant_ratio=0.5, merge_cluster=False):
    if face_fragment_path is None:
        face_fragment_path = utils.generate_output_path('./data/cluster', video_path)

    batch_size = 200
    image_size = 160

    # Load classifier
    classifier_filename = os.path.expanduser(classifier_path)
    with open(classifier_filename, 'rb') as f:
        (classifier, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename)

    # this has to be run AFTER export_mappingfile
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            with open('data/tracker_saved_greater_5.txt') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            time_dict = dict()
            for cluster_id in content:
                id_cluster = int(cluster_id.split('.')[0])
                id_frame = int(cluster_id.split('.')[1])
                time_dict.update({id_cluster: id_frame})

            # Get the path of the facenet model and load it
            facenet.load_model(facenet_model_path)

            dataset = facenet.get_dataset(face_fragment_path)
            paths, cur_labels = facenet.get_image_paths_and_labels(dataset)
            label_name = [cls.name.replace('_', ' ') for cls in dataset]
            cur_labels = np.array([int(label_name[i]) for i in cur_labels])

            print('Number of cluster: {}'.format(len(dataset)))
            print('Number of images: {}'.format(len(paths)))
            print('Loading feature extraction model')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            cur_embs = np.zeros((nrof_images, embedding_size))
            # cur_embs contains the embeddings of the faces found in the current video

            for i in range(nrof_batches):
                print('%d/%d' % (i + 1, nrof_batches))
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                cur_embs[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            train_embs = pd.read_csv('data/embedding/embedding.csv', header=None)
            train_embs = np.array(train_embs.values)
            train_labels = pd.read_csv('data/embedding/label.csv', header=None, names=["label", "path"])
            train_labels = np.array(train_labels['label'].values)
            # train_embs contains the embeddings of the faces used for training

            print('Start testing...')
            predictions = classifier.predict_proba(cur_embs)

            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            # FIXME write in a more elegant way
            index_great = [idx for idx, val in enumerate(best_class_probabilities) if val > prob_threshold]
            prediction_new = [val for idx, val in enumerate(best_class_indices) if idx in index_great]
            labels_new = [val for idx, val in enumerate(cur_labels) if idx in index_great]

            # group prediction in a cluster dictionary
            cluster_result = {}
            for a, b in zip(labels_new, prediction_new):
                cluster_result.setdefault(a, []).append(b)

            # voting system
            interest_cluster = {}
            for key, value in cluster_result.items():
                try:
                    dominant = mode(value)
                    if value.count(dominant) / len(value) * 1.0 > dominant_ratio:
                        name = dominant
                        # TODO save the confidence?

                except StatisticsError:  # no clear winner
                    # name = "-1"
                    pass

                interest_cluster.update({key: name})

            known_persons = list(set([j for i, j in interest_cluster.items()]))

            for person_id in known_persons:
                person = class_names[person_id]
                person_clusters = [i for i, j in interest_cluster.items() if j == person_id]

                if merge_cluster:
                    person_clusters = merge_consecutive_clusters(person_clusters, face_fragment_path)

                print("{}: {}".format(person, person_clusters))
                if person_clusters:
                    count_frames_for_cluster(person_clusters, face_fragment_path)
                    for cluster_id in person_clusters:
                        # get the training faces of the current person
                        emb_array_training = train_embs[np.nonzero(train_labels == person_id)]
                        # get the video faces of the current cluster
                        emb_array_testing = cur_embs[np.nonzero(cur_labels == cluster_id)]

                        distances = cluster_distance(emb_array_testing, emb_array_training)
                        distances.sort(reverse=True)
                        mean_of_best_3 = mean(distances[0:3])  # TODO why?
                        print("Cluster %s - %s - Cosine distance Mean: %f" % (cluster_id, person, mean_of_best_3))
                        print("-> appears at frame %s" % time_dict[int(cluster_id)])


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


def merge_consecutive_clusters(clusters, face_fragment_path):
    clusters.sort()

    for i in clusters:
        for j in clusters:
            if i < j:
                if i == j + 1:
                    src_dir = os.path.join(face_fragment_path, j)
                    dst_dir = os.path.join(face_fragment_path, i)
                    merge_folders(src_dir, dst_dir)
                    clusters.remove(j)
                    break
    return clusters


def count_frames_for_cluster(person_clusters, directory):
    for i in person_clusters:
        return len(os.listdir(os.path.join(directory, str(i))))


def cluster_distance(a, b):
    return [(1 - distance.cosine(i, j)) for i, j in zip(a, b)]


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument("--threshold", type=float, help='threshold',
                        default=0.7)
    parser.add_argument('--face_fragment_path', type=str,
                        help='Path for saving the fragment of image containing the detected face.\n'
                             'By default is in `data\\cluster\\<video_name>`')
    parser.add_argument('--classifier_path', type=str, help='Path to KNN classifier',
                        default="classifier/classifier.pkl")
    parser.add_argument('--model_path', type=str, help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--dominant_ratio', type=float,
                        help='Ratio threshold to decide cluster name', default=0.5)
    parser.add_argument('--merge_cluster', default=False, action='store_true',
                        help='Include the argument for merging the clusters')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.video, args.face_fragment_path, args.classifier_path, args.model_path, args.threshold,
         args.dominant_ratio, args.merge_cluster)
