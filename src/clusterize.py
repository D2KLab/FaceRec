import argparse
import os
import shutil
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
from scipy.spatial import distance

from .utils import utils

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")


# IMPORTANT: this has to be run AFTER the tracker

def main(video_path, tracker_path, confidence_threshold=0.7, dominant_ratio=0.5, merge_cluster=False):
    if tracker_path is None:
        tracker_path = utils.generate_output_path('./data/out', video_path)

    # setup all paths
    # cluster_path = os.path.join(tracker_path, 'cluster')
    predictions_csv = os.path.join(tracker_path, 'predictions.csv')

    predictions_load = pd.read_csv(predictions_csv,
                                   dtype={'x1': np.int, 'y1': np.int, 'x2': np.int, 'y2': np.int,
                                          'track_id': np.int, 'frame': np.int, 'tracker_sample': np.int})
    # print(predictions_load.head())

    # START ALGORITHM
    interest_cluster = {}
    for track in predictions_load.track_id.unique():
        involved = predictions_load[predictions_load.track_id == track]
        involved = involved[involved.confidence >= confidence_threshold]
        predicted = involved.name.values.tolist()
        name = ""
        try:
            dominant = mode(predicted)  # TODO normalise with confidence
            if predicted.count(dominant) / float(len(predicted)) > dominant_ratio:
                name = dominant
        except StatisticsError:  # no clear winner
            pass

        interest_cluster.update({track: name})

    known_persons = list(set([j for i, j in interest_cluster.items() if j]))

    for person in known_persons:
        person_clusters = [i for i, j in interest_cluster.items() if j == person]
        print("* {}: {}".format(person, person_clusters))

        if merge_cluster:
            involved = predictions_load[predictions_load.track_id.isin(person_clusters)]

            # select the max and min sample (for the merging)
            person_clusters = []
            for inv in involved.track_id.unique():
                x = involved[involved.track_id == inv]
                max = x.tracker_sample.max()
                min = x.tracker_sample.min()

                person_clusters.append({'id': inv, 'max': max, 'min': min})

            person_clusters = merge_consecutive_clusters(person_clusters, tracker_path)
            print("---> {}: {}".format(person, [c['id'] for c in person_clusters]))

        # COSINE SIMILARITY
        # I don't see any utility now, but maybe in the future

        # train_embs contains the embeddings of the faces used for training
        # train_embs = pd.read_csv('data/embedding/embedding.csv', header=None)
        # train_embs = np.array(train_embs.values)
        # train_labels = pd.read_csv('data/embedding/label.csv', header=None, names=["label", "path"])
        # train_labels = np.array(train_labels['label'].values)

        # for cluster_id in person_clusters:
        #     # get the training faces of the current person
        #     emb_array_training = train_embs[np.nonzero(train_labels == person_id)]
        #     # get the video faces of the current cluster
        #     emb_array_testing = cur_embs[np.nonzero(cur_labels == cluster_id)]
        #
        #     distances = cluster_distance(emb_array_testing, emb_array_training)
        #     distances.sort(reverse=True)
        #     mean_of_best_3 = mean(distances[0:3])  # TODO why?
        #     print("Cluster %s - %s - Cosine distance Mean: %f" % (cluster_id, person, mean_of_best_3))


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
    for i in clusters:
        for j in clusters:
            # if j['min'] - i['max'] < 5:
            #     print('%d : %d > %d : %d' % (i['max'], j['min'], i['id'], j['id']))
            if j['min'] - i['max'] == 1:
                src_dir = os.path.join(face_fragment_path, j['id'])
                dst_dir = os.path.join(face_fragment_path, i['id'])
                merge_folders(src_dir, dst_dir)
                clusters.remove(j)
                break
    return clusters


def cluster_distance(a, b):
    return [(1 - distance.cosine(i, j)) for i, j in zip(a, b)]


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='The threshold for the classification confidence')
    parser.add_argument('--tracker_path', type=str,
                        help='Output path of the tracker script.\n'
                             'By default is in `data\\out\\<video_name>`')
    parser.add_argument('--dominant_ratio', type=float,
                        help='Ratio threshold to decide cluster name', default=0.5)
    parser.add_argument('--merge_cluster', default=False, action='store_true',
                        help='Include the argument for merging the clusters')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.video, args.tracker_path, args.confidence_threshold,
         args.dominant_ratio, args.merge_cluster)
