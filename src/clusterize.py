import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import mode
from sklearn.utils.extmath import weighted_mode

from .utils.utils import rect2xywh, generate_output_path


# IMPORTANT: this has to be run AFTER the tracker

def get_avg_rect(rects):
    rects = np.array(rects)
    x1 = min(rects[:, 0])
    y1 = min(rects[:, 1])
    x2 = max(rects[:, 2])
    y2 = max(rects[:, 3])
    return [x1, y1, x2, y2]


def update_rect_in(previous_cluster, rects):
    avg_rect = get_avg_rect([r for r in rects])
    previous_cluster['rect'] = avg_rect
    previous_cluster['bounding'] = rect2xywh(*avg_rect)


# predictions is a pandas dataframe
def main(predictions, confidence_threshold=0.7, dominant_ratio=0.6, weighted_dominant_ratio=0.4, merge_cluster=False,
         min_length=1):
    if len(predictions) < 1:
        return []
    predictions = predictions.sort_values(by=['track_id', 'tracker_sample'])
    # filter out tracks with less than 3 records
    stat = predictions.groupby('track_id').size().to_frame('size')
    good_ids = [i for i, s in stat.iterrows()]
    predictions = predictions[predictions['track_id'].isin(good_ids)]

    # START ALGORITHM
    interest_cluster = {}
    for track in predictions.track_id.unique():
        involved = predictions[predictions.track_id == track]
        confidences = involved.confidence.values.tolist()
        predicted = involved.name.values.tolist()
        name = ""
        dominant, count = weighted_mode(predicted, confidences)
        dominant2, count2 = mode(predicted)
        if count[0] / float(len(predicted)) > weighted_dominant_ratio \
                and dominant[0] == dominant2[0] and count2[0] / float(len(predicted)) > dominant_ratio:
            name = dominant[0]
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

            if merge_cluster and previous_cluster is not None and min - previous_cluster['end_sample'] == 1:
                # merge here
                duration_prev = float(previous_cluster['end_sample'] - previous_cluster['start_sample'] + 1)
                previous_cluster['end_sample'] = max
                previous_cluster['end_frame'] = x.frame.max()
                previous_cluster['end_npt'] = x.npt.max()
                rc = x.rect.values.tolist()
                rc.append(previous_cluster['rect'])
                update_rect_in(previous_cluster, rc)

                duration_cur = float(max - min + 1)
                confidence_prev = previous_cluster['confidence']
                confidence = x[x.name == person].confidence.mean()
                previous_cluster['confidence'] = ((confidence_prev * duration_prev) + (
                        confidence * duration_cur)) / (duration_prev + duration_cur)
                previous_cluster['merged_tracks'].append(int(x.track_id.iloc[0]))
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
            confidence = x[x.name == person].confidence
            previous_cluster['confidence'] = confidence.mean()
            previous_cluster['name'] = person
            previous_cluster['merged_tracks'] = [int(previous_cluster['track_id'])]

            update_rect_in(previous_cluster, x.rect.values.tolist())

            del previous_cluster['npt']
            del previous_cluster['frame']
            del previous_cluster['tracker_sample']
            if '_id' in previous_cluster:
                del previous_cluster['_id']

        if previous_cluster is not None:  # last cluster
            person_clusters.append(previous_cluster['track_id'])
            final_clusters.append(previous_cluster)

        # print("* {}: {}".format(person, person_clusters))

    final_clusters = [s for s in final_clusters
                      if longer_than(min_length, s) and s['confidence'] >= confidence_threshold]
    return sanitize(final_clusters)


def unknown_clusterise(feat_clusters, assigned_tracks, raw_tracks):
    assigned_tracks = [item for sublist in assigned_tracks for item in sublist]

    tracks_done = []
    for clus in feat_clusters:
        elems = clus['elements']
        if any([x['track'] in assigned_tracks for x in elems]):
            continue

        for e in elems:
            tr = e['track']
            if tr in tracks_done:
                continue

            raw_tracks.loc[raw_tracks.track_id == tr, 'name'] = f'Unknown {clus["id"]}'
            tracks_done.append(tr)

    tracks = raw_tracks[raw_tracks.track_id.isin(tracks_done)]
    tracks = main(tracks, confidence_threshold=0, dominant_ratio=0, weighted_dominant_ratio=0, merge_cluster=True,
                  min_length=1)
    total_time = {}
    for t in tracks:
        del t['confidence']
        t['unknown'] = True
        if '_id' in t:
            del t['_id']

        time = t['end_npt'] - t['start_npt']
        if t['name'] not in total_time:
            total_time[t['name']] = time
        else:
            total_time[t['name']] += time

    output = [t for t in tracks if total_time[t['name']] > 5]

    return output


def longer_than(length, cluster):
    duration = cluster['end_sample'] - cluster['start_sample']
    return duration >= length


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def sanitize(input):
    j = json.dumps(input, default=convert)
    return json.loads(j)


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
        tracker_path = generate_output_path('./data/out', args.project, args.video_path)
    else:
        tracker_path = args.tracker_path

    # setup all paths
    predictions_csv = os.path.join(tracker_path, 'predictions.csv')

    predictions_load = pd.read_csv(predictions_csv,
                                   dtype={'x1': int, 'y1': int, 'x2': int, 'y2': int,
                                          'track_id': int, 'frame': int, 'tracker_sample': int})

    main(predictions_load, args.confidence_threshold, args.dominant_ratio, args.merge_cluster)
