import datetime
import os
import json
import time
from threading import Thread

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_restx import Api, Resource
from werkzeug.middleware.proxy_fix import ProxyFix

from src import *
from src.connectors import antract_connector as antract
from src.utils import utils, uri_utils
from bson.json_util import dumps

TRAINING_IMG = 'data/training_img_aligned/'

IMG_DIR = os.path.join(os.getcwd(), TRAINING_IMG)
VIDEO_DIR = os.path.join(os.getcwd(), 'video')

os.makedirs('database', exist_ok=True)

database.init()

flask_app = Flask(__name__)
flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_port=1, x_for=1, x_host=1, x_prefix=1)
api = Api(app=flask_app,
          version="0.1.0",
          default='facerec',
          title="Face Recognition Api",
          description="Recognise celebrities on videos.", )
CORS(flask_app)

PROJECTS = [p for p in os.listdir(TRAINING_IMG) if os.path.isdir(os.path.join(TRAINING_IMG, p))]
project_param = {'description': 'The project context of the call', 'enum': PROJECTS, 'required': True}


def now():
    return datetime.datetime.now().isoformat()


@api.route('/projects')
@api.doc(description="Get list of active projects.")
class Projects(Resource):
    def get(self):
        return jsonify(PROJECTS)


@api.route('/training-set')
@api.doc(description="Get list of training images with classes.",
         params={'project': project_param})
class TrainingSet(Resource):
    def get(self):
        dataset = request.args.get('project', 'general')
        folder = os.path.join(TRAINING_IMG, dataset)

        labels, paths = utils.fetch_dataset(folder)
        results = {}
        for path, c in zip(paths, labels):
            path = path.replace(TRAINING_IMG, 'training_img_aligned/')
            if c not in results:
                results[c] = {
                    'class': c,
                    'path': [path]
                }
            else:
                results[c]['path'].append(path)
        return jsonify(list(results.values()))


# http://127.0.0.1:5000/crawler?project=antract&q=Charles De Gaulle;Vincent Auriol;Pierre Mendès France;Georges Bidault;Guy Mollet;François Mitterrand;Georges Pompidou;Elisabeth II;Konrad Adenauer;Dwight Eisenhower;Nikita Khrouchtchev;Viatcheslav Molotov;Ahmed Ben Bella
# http://127.0.0.1:5000/crawler?project=memad&q=Annastiina Heikkilä;Frans Timmermans;Manfred Weber;Markus Preiss;Ska Keller;Emilie Tran Nguyen;Jan Zahradil;Margrethe Vestager;Nico Cué;Laura Huhtasaari;Asseri Kinnunen
@api.route('/crawler')
@api.doc(
    description="Search faces of people in the web to be added to the dataset.",
    params={
        'q': {
            'required': True,
            'description': 'The name of the person, or multiple individuals separated by a semicolon, '
                           'like in "Tom Hanks;Monica Bellucci"'},
        'project': project_param
    })
class Crawler(Resource):
    def get(self):
        start_time = time.time()

        q = request.args.get('q')
        if q is None:
            raise ValueError('Missing required parameter: q')

        project = request.args.get('project', default='general')

        for keyword in q.split(';'):
            crawler.main(keyword, max_num=30, project=project)

        return jsonify({
            'task': 'crawl',
            'time': now(),
            'execution_time': (time.time() - start_time),
            'status': 'ok'
        })


@api.route('/train/<string:project>')
@api.doc(description="Trigger the training of the model")
class Training(Resource):
    def get(self, project):
        start_time = time.time()

        classifier.main(classifier='SVM', project=project, discard_disabled="true")
        return jsonify({
            'task': 'train',
            'time': now(),
            'execution_time': (time.time() - start_time),
            'status': 'ok'
        })


# http://127.0.0.1:5000/track?speedup=25&video=video/yle_a-studio_8a3a9588e0f58e1e40bfd30198274cb0ce27984e.mp4
# http://127.0.0.1:5000/track?format=ttl&video=http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
@api.route('/track')
@api.doc(description="Extract from the video all the continuous positions of the people in the dataset",
         params={
             'video': {'required': True, 'description': 'URI of the video to be analysed'},
             'project': project_param,
             'speedup': {'default': 25, 'type': int,
                         'description': 'Number of frame to wait between two iterations of the algorithm'},
             'no_cache': {'type': bool, 'default': False,
                          'description': 'Set it if you want to recompute the annotations'},
             'format': {'default': 'json', 'enum': ['json', 'ttl'], 'description': 'Set the output format'},
         })
class Track(Resource):
    def get(self):
        video_id = request.args.get('video').strip()
        project = request.args.get('project').strip()
        speedup = request.args.get('speedup', type=int, default=25)
        no_cache = 'no_cache' in request.args.to_dict() and request.args.get('no_cache') != 'false'

        video = None
        locator = video_id
        if not no_cache:
            video = database.get_all_about(video_id, project)
            if video:
                locator = video['locator']

        need_run = not video or 'tracks' not in video and video.get('status') != 'RUNNING'
        if not video or need_run:
            if video_id.startswith('http'):  # it is a uri!
                locator, video = uri_utils.uri2video(video_id)
                video_id = video['locator']
            elif not os.path.isfile(video_id):
                raise FileNotFoundError('video not found: %s' % video_id)
            else:
                video = {'locator': video_id}
            database.save_metadata(video)

        if need_run:
            database.clean_analysis(video_id, project)
            database.save_status(video_id, project, 'RUNNING')
            video['status'] = 'RUNNING'
            Thread(target=run_tracker, args=(locator, speedup, video_id, project)).start()
        elif 'tracks' in video and len(video['tracks']) > 0:
            raw_tracks = clusterize.from_dict(video['tracks'])

            video['tracks'] = clusterize.main(raw_tracks, confidence_threshold=0, merge_cluster=True)
            assigned_tracks = [t['merged_tracks'] for t in video['tracks']]
            if 'feat_clusters' in video:
                video['feat_clusters'] = clusterize.unknown_clusterise(video['feat_clusters'], assigned_tracks,
                                                                       raw_tracks)

        if '_id' in video:
            del video['_id']  # the database id should not appear on the output

        fmt = request.args.get('format')
        if fmt == 'ttl':
            return Response(semantifier.semantify(video), mimetype='text/turtle')
        return jsonify(video)


def run_tracker(video_path, speedup, video, project):
    try:
        return tracker.main(video_path, project=project, video_speedup=speedup, export_frames=True, video_id=video)
    except RuntimeError:
        database.save_status(video, project, 'ERROR')


@flask_app.route('/get_locator')
def send_video():
    path = request.args.get('video')

    if path.startswith('http'):
        video_path, info = uri_utils.uri2video(path)
        return video_path
    else:
        return send_from_directory(VIDEO_DIR, path, as_attachment=True)


@flask_app.route('/get_metadata')
def get_metadata():
    path = request.args.get('video')

    if path.startswith('http://www.ina.fr/'):
        return jsonify(antract.get_metadata_for(path)[0])
    else:
        return None


@flask_app.route('/appearance/<string:person>')
def get_appearances(person):
    project = request.args.get('project')

    return jsonify(json.loads(dumps(database.get_video_with(person, project))))


@flask_app.route('/training_img_aligned/<path:subpath>')
def send_img(subpath=None):
    dirname = os.path.dirname(subpath)
    filename = os.path.basename(subpath)
    return send_from_directory(os.path.join(IMG_DIR, dirname), filename, as_attachment=True)


@api.route('/disabled/<string:project>')
class Disabled(Resource):
    def get(self, project):
        DISABLED_FILE = os.path.join(TRAINING_IMG, project, 'disabled.txt')

        if not os.path.isfile(DISABLED_FILE):
            # automatic disable
            return jsonify(classifier.get_outlier_list(project))

        with open(DISABLED_FILE) as f:
            dis = f.read().split('\n')
            return jsonify(dis)

    def post(self, project):
        data = request.json
        DISABLED_FILE = os.path.join(TRAINING_IMG, project, 'disabled.txt')

        with open(DISABLED_FILE, 'w') as f:
            for x in data:
                f.write(x)
                f.write('\n')
            f.close()

        return 'ok'


@api.errorhandler(ValueError)
def handle_invalid_usage(error):
    response = jsonify({
        'status': 'error',
        'error': str(error),
        'time': now()
    })
    response.status_code = 422
    return response


if __name__ == '__main__':
    flask_app.run()
