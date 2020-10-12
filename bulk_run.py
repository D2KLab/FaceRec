import os

import argparse
import json
import sys

import pandas as pd
from tqdm import tqdm

from src import clusterize, database
from src.tracker import Tracker
from src.utils import uri_utils

os.makedirs('database', exist_ok=True)

database.init()


# 'evaluation/dataset_memad.csv'
# 'memad-gt'
def main(input, project, skip_tracking=False):
    if not skip_tracking:
        # TODO check if input is a csv or a folder
        df = pd.read_csv(input)
        tr = Tracker(project=project)

        all_results = []
        v = None
        old = None
        for i, x in tqdm(df.iterrows(), total=len(df)):
            if 'type' in x and x['type'] != 'VIDEO':
                continue
            start = int(x['start']) if 'start' in x else None
            end = int(x['end']) if 'end' in x else None
            fragment = f'{start},{end + 1}' if start is not None else None

            if 'media' in x:
                media = x['media']
                video_id = x['media']
                if media != old:
                    v, metadata = uri_utils.uri2video(media)
                    database.save_metadata(metadata)
            else:
                v = '/data/memad-uc22/' + x['Name']
                video_id = x['kgURI']
                _, metadata = uri_utils.uri2video(video_id)
                database.save_metadata(metadata)

            database.clean_analysis(video_id, project)
            database.save_status(video_id, project, 'RUNNING')
            res = tr.run(v, export_frames=True, fragment=fragment, video_id=video_id, verbose=False)
            all_results.append(res)
        with open(f'results_{project}.json', 'w') as f:
            json.dump(all_results, f)

    else:
        with open(f'results_{project}.json', 'r') as f:
            all_results = json.load(f)

    clusters = []
    for r in all_results:
        c = clusterize.main(clusterize.from_dict(r), dominant_ratio=0.6, weighted_dominant_ratio=0.4,
                            confidence_threshold=0.6, merge_cluster=True, min_length=1)
        clusters.append(c)

    with open(f'results_{project}_clusters.json', 'w') as f:
        json.dump(clusters, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The csv containing the videos to analyse.')
    parser.add_argument('--project', type=str, default='general',
                        help='Name of the collection to be part of')
    parser.add_argument('--skip_tracking', action='store_true', default=False,
                        help='Only recompute clustering')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.input, args.project, args.skip_tracking)

# python bulk_run.py -i evaluation/dataset_antract.csv --project antract
