import os
from src.utils.uri_utils import uri2video, clean_locator
from src import database, tracker, clusterize
import time
import json

project = 'antract_full'
database.init()

input_dir = '/data/antract_corpus'
output_dir = '/data/antract_corpus/processed/'
os.makedirs(output_dir, exist_ok=True)

all_videos = [x for x in os.listdir(input_dir) if ".mp4" in x][0:1]

for i, video_name in enumerate(all_videos):
    print(f'%%%%%%%%%% Processing {i} of {len(all_videos)}')
    video_file = os.path.join(input_dir, video_name)
    video_id = video_name.replace('.mp4', '').strip()

    media = 'http://www.ina.fr/media/' + video_id
    print(media)

    v = database.get_all_about(media, project)
    tracks = [] if (not v or 'tracks' not in v) else v['tracks']
    need_run = not v or (len(tracks) < 1 and v.get('status') != 'RUNNING')
    if need_run:
        locator, v = uri2video(media)
        v['locator'] = clean_locator(locator)
        database.save_metadata(v)

        database.clean_analysis(locator, project)
        database.save_status(locator, project, 'RUNNING')
        try:
            time1 = time.time()
            tracker.main(video_file, project=project, video_speedup=25, export_frames=False)
            v = database.get_all_about(media, project)

            time2 = time.time()
            print('{:s} processed in {:.3f} s'.format(media, (time2 - time1)))
        except RuntimeError:
            database.save_status(locator, project, 'ERROR')

    raw_tracks = clusterize.from_dict(v['tracks'])

    v['tracks'] = clusterize.main(raw_tracks, confidence_threshold=0, merge_cluster=True)
    assigned_tracks = [t['merged_tracks'] for t in v['tracks']]
    if 'feat_clusters' in v:
        v['feat_clusters'] = clusterize.unknown_clusterise(v['feat_clusters'], assigned_tracks, raw_tracks)

    if '_id' in v:
        del v['_id']  # the database id should not appear on the output

    with open(os.path.join(output_dir, video_id + '.json'), 'w') as fp:
        json.dump(dict, fp)

print('completed')
