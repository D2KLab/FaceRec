import argparse
import os
import json
import shutil
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
from scipy.spatial import distance

from .utils import utils


# IMPORTANT: this has to be run AFTER the tracker

# predictions is a pandas dataframe
def get_avg_rect(rects):
    rects = np.array(rects.values.tolist())
    x1 = min(rects[:, 0])
    y1 = min(rects[:, 1])
    x2 = max(rects[:, 2])
    y2 = max(rects[:, 3])
    return [x1, y1, x2, y2]


def main(predictions, confidence_threshold=0.7, dominant_ratio=0.5, merge_cluster=False, min_length=1):
    predictions = predictions.sort_values(by=['track_id', 'tracker_sample'])
    # filter out tracks with less than 3 records
    stat = predictions.groupby('track_id').size().to_frame('size')
    good_ids = [i for i, s in stat.iterrows() if s['size'] >= min_length]
    predictions = predictions[predictions['track_id'].isin(good_ids)]

    # START ALGORITHM
    interest_cluster = {}
    for track in predictions.track_id.unique():
        involved = predictions[predictions.track_id == track]
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
    final_clusters = []
    for person in known_persons:
        person_clusters = [i for i, j in interest_cluster.items() if j == person]

        involved = predictions[predictions.track_id.isin(person_clusters)]
        person_clusters = []
        previous_cluster = None

        for id in involved.track_id.unique():
            x = involved[involved.track_id == id]
            # select the max and min sample (for the merging)
            max = x.tracker_sample.max()
            min = x.tracker_sample.min()

            if merge_cluster and previous_cluster is not None and previous_cluster['end_sample'] - min == 1:
                # merge here
                previous_cluster['end_sample'] = max
                previous_cluster['end_frame'] = x.frame.max()
                previous_cluster['end_npt'] = x.npt.max()
                continue
                # in case, merge the folders
            elif previous_cluster is not None:
                final_clusters.append(previous_cluster)
                person_clusters.append(previous_cluster['track_id'])

            previous_cluster = x.to_dict('records')[0]
            previous_cluster['end_sample'] = max
            previous_cluster['start_sample'] = min
            previous_cluster['end_frame'] = x.frame.max()
            previous_cluster['start_frame'] = x.frame.min()
            previous_cluster['end_npt'] = x.npt.max()
            previous_cluster['start_npt'] = x.npt.min()
            previous_cluster['confidence'] = x.confidence.mean()  # FIXME give a smarter confidence
            previous_cluster['name'] = person

            avg_rect = get_avg_rect(x.rect)
            previous_cluster['rect'] = avg_rect
            previous_cluster['bounding'] = utils.rect2xywh(*avg_rect)

            del previous_cluster['npt']
            del previous_cluster['frame']
            del previous_cluster['tracker_sample']
            del previous_cluster['_id']

        if previous_cluster is not None:
            person_clusters.append(previous_cluster['track_id'])
            final_clusters.append(previous_cluster)

        # print("* {}: {}".format(person, person_clusters))

    return sanitize(final_clusters)


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def sanitize(input):
    j = json.dumps(input, default=convert)
    return json.loads(j)

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
    parser.add_argument('--project', type=str, default='general',
                        help='Name of the collection to be part of')
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


def from_dict(input):
    return pd.DataFrame(input)


if __name__ == '__main__':
    args = parse_args()

    if args.tracker_path is None:
        tracker_path = utils.generate_output_path('./data/out', args.project, args.video_path)
    else:
        tracker_path = args.tracker_path

    # setup all paths
    # cluster_path = os.path.join(tracker_path, 'cluster')
    predictions_csv = os.path.join(tracker_path, 'predictions.csv')

    predictions_load = pd.read_csv(predictions_csv,
                                   dtype={'x1': int, 'y1': int, 'x2': int, 'y2': int,
                                          'track_id': int, 'frame': int, 'tracker_sample': int})

    main(predictions_load, args.confidence_threshold, args.dominant_ratio, args.merge_cluster)
