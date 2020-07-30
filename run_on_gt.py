import json
import pandas as pd
from src.tracker import Tracker
from src import clusterize
from tqdm import tqdm
from src.connectors import limecraft_connector as limecraft

df = pd.read_csv('evaluation/dataset_memad.csv')
tr = Tracker(project='memad-gt')


all_results = []
for i, x in tqdm(df.iterrows(), total=len(df)):
    start = int(x['start'])
    end = int(x['end'])
    locator = x['locator']
    v = limecraft.locator2video(locator)
    print(v)
    res = tr.run(v, export_frames=True, fragment=f'{start},{end}', video_id=x['media'])
    print(res)
    all_results.append(res)


with open('results_memad_gt.json', 'w') as f:
    json.dump(all_results, f)


clusters  = []
for r in all_results:
    c = clusterize.main(clusterize.from_dict(r), confidence_threshold=0.0, merge_cluster=True, min_length=1)
    clusters.append(c)
    
with open('results_memad_gt_clusters.json', 'w') as f:
    json.dump(clusters, f)

