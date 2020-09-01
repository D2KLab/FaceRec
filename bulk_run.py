import sys
import argparse
import json
import pandas as pd
from src.tracker import Tracker
from src import clusterize
from tqdm import tqdm
from src.utils import uri_utils


# 'evaluation/dataset_memad.csv'
# 'memad-gt'
def main(input, project):
    # TODO check if input is a csv or a folder
    df = pd.read_csv(input)
    tr = Tracker(project=project)

    all_results = []
    v = None
    old = None
    for i, x in tqdm(df.iterrows(), total=len(df)):
        start = int(x['start']) if 'start' in x else None
        end = int(x['end']) if 'end' in x else None
        fragment = f'{start},{end}' if start is not None else None
        media = x['media']
        if media != old:
            v, metadata = uri_utils.uri2video(media)

        res = tr.run(v, export_frames=True, fragment=fragment, video_id=x['media'])
        all_results.append(res)

    with open(f'results_{project}.json', 'w') as f:
        json.dump(all_results, f)

    clusters = []
    for r in all_results:
        c = clusterize.main(clusterize.from_dict(r), confidence_threshold=0.0, merge_cluster=True, min_length=1)
        clusters.append(c)

    with open(f'results_{project}_clusters.json', 'w') as f:
        json.dump(clusters, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The csv containing the videos to analyse.')
    parser.add_argument('--project', type=str, default='general',
                        help='Name of the collection to be part of')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.input, args.project)
