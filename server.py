import os
import time
import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_restplus import Api, Resource
from threading import Thread

from src import *
from src.utils import utils
from src.connectors import antract_connector as antract

TRAINING_IMG = 'data/training_img/'

IMG_DIR = os.path.join(os.getcwd(), TRAINING_IMG)
VIDEO_DIR = os.path.join(os.getcwd(), 'video')

os.makedirs('database', exist_ok=True)

database.init()

flask_app = Flask(__name__)
api = Api(app=flask_app,
          version="0.1.0",
          title="Face Recognition Api",
          description="Recognise celebrities on videos.", )
CORS(flask_app)


def now():
    return datetime.datetime.now().isoformat()


@api.route('/projects')
@api.doc(description="Get list of active projects.")
class Projects(Resource):
    def get(self):
        return jsonify([p for p in os.listdir(TRAINING_IMG) if os.path.isdir(os.path.join(TRAINING_IMG, p))])


@api.route('/training-set')
@api.doc(description="Get list of training images with classes.",
         params={'project': 'The project those images belong to'})
class TrainingSet(Resource):
    def get(self):
        dataset = request.args.get('project', 'general')
        folder = os.path.join(TRAINING_IMG, dataset)

        labels, paths = utils.fetch_dataset(folder)
        results = {}
        for path, c in zip(paths, labels):
            path = path.replace(TRAINING_IMG, 'training_img/')
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
        'project': 'The project those images belong to'
    })
class Crawler(Resource):
    def get(self):
        start_time = time.time()

        q = request.args.get('q')
        if q is None:
            raise ValueError('Missing required parameter: q')

        project = request.args.get('project', default='general')

        for keyword in q.split(';'):
            crawler.main(keyword, max_num=100, project=project)

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

        FaceDetector.main(project=project)
        classifier.main(classifier='SVM', project=project)
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
             'project': {'required': True, 'description': 'The collection of people to detect'},
             'speedup': {'default': 25, 'type': int,
                         'description': 'Number of frame to wait between two iterations of the algorithm'},
             'no_cache': {'type': bool, 'default': False,
                          'description': 'Set it if you want to recompute the annotations'},
             'format': {'default': 'json', 'enum': ['json', 'ttl'], 'description': 'Set the output format'},
         })
class Track(Resource):
    def get(self):
        video = request.args.get('video')
        project = request.args.get('project')
        speedup = request.args.get('speedup', type=int, default=25)
        no_cache = 'no_cache' in request.args.to_dict() and request.args.get('no_cache') != 'false'

        v = None
        video_path = video
        if not no_cache:
            v = database.get_all_about(video, project)
            if v:
                video_path = v['locator']

        need_run = not v or 'tracks' not in v and v.get('status') != 'RUNNING'
        if not v or need_run:
            if video.startswith('http'):  # it is a uri!
                video_path, v = utils.uri2video(video)
                video = utils.clean_locator(v['locator'])
            elif not os.path.isfile(video):
                raise FileNotFoundError('video not found: %s' % video)
            else:
                v = {'locator': video}
            database.save_metadata(v)

        if need_run:
            database.clean_analysis(video, project)
            database.save_status(video, project, 'RUNNING')
            v['status'] = 'RUNNING'
            Thread(target=run_tracker, args=(video_path, speedup, video, project)).start()
        elif 'tracks' in v and len(v['tracks']) > 0:
            v['tracks'] = clusterize.main(clusterize.from_dict(v['tracks']),
                                          confidence_threshold=0, merge_cluster=True)

        if '_id' in v:
            del v['_id']  # the database id should not appear on the output

        fmt = request.args.get('format')
        if fmt == 'ttl':
            return Response(semantifier.semantify(v), mimetype='text/turtle')
        return jsonify(v)


def run_tracker(video_path, speedup, video, project):
    try:
        return tracker.main(video_path, project=project, video_speedup=speedup, export_frames=True)
    except RuntimeError:
        database.save_status(video, project, 'ERROR')


# # http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
# # http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/eurovaalit-2019-kuka-johtaa-eurooppaa/0460c1b7d735e3fc796aa2829811aa1ae5dc9fa8
# # http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/eurovaalit-2019-kuka-johtaa-eurooppaa/d9d05488b35db559cdef35bac95f518ee0dda76a
# # http://127.0.0.1:5000/recognise?speedup=50&format=ttl&no_cache&video=http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
# @api.route('/recognise')
# @api.doc(description="Extract from each frame of the video the positions of the people in the dataset",
#          params={
#              'video': {'required': True, 'description': 'URI of the video to be analysed'},
#              'speedup': {'default': 25, 'type': int,
#                          'description': 'Number of frame to wait between two iterations of the algorithm'},
#              'no_cache': {'type': bool, 'default': False,
#                           'description': 'Set it if you want to recompute the annotations'},
#              'format': {'default': 'json', 'enum': ['json', 'ttl'], 'description': 'Set the output format'}
#          })
# class Recognise(Resource):
#     def get(self):
#         start_time = time.time()
#
#         video = request.args.get('video')
#         speedup = request.args.get('speedup', type=int, default=25)
#         no_cache = 'no_cache' in request.args.to_dict()
#
#         results = None
#         info = None
#         if not no_cache:
#             results = db_detection.search(Query().video == video)
#             if results and len(results) > 0:
#                 results = results[0]
#
#         if not results:
#             video_path = video
#             if video.startswith('http'):  # it is a uri!
#                 video_path, info = utils.uri2video(video)
#             elif not os.path.isfile(video):
#                 raise FileNotFoundError('video not found: %s' % video)
#
#             r = FaceRecogniser.main(video_path, video_speedup=speedup, confidence_threshold=0.2)
#             results = {
#                 'task': 'recognise',
#                 'status': 'ok',
#                 'execution_time': (time.time() - start_time),
#                 'time': now(),
#                 'video': video,
#                 'info': info,
#                 'results': r
#             }
#
#             # TODO insert aliases in the cache
#             # delete previous results
#             db_detection.remove(where('video') == video)
#             db_detection.insert(results)
#
#         # with open('recognise.json', 'w') as outfile:
#         #     json.dump(r, outfile)
#         fmt = request.args.get('format')
#         if fmt == 'ttl':
#             return Response(semantifier.semantify(results), mimetype='text/turtle')
#
#         return jsonify(results)


@flask_app.route('/get_locator')
def send_video():
    path = request.args.get('video')

    if path.startswith('http'):
        video_path, info = utils.uri2video(path)
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


@flask_app.route('/training_img/<path:subpath>')
def send_img(subpath=None):
    dirname = os.path.dirname(subpath)
    filename = os.path.basename(subpath)
    return send_from_directory(os.path.join(IMG_DIR, dirname), filename, as_attachment=True)


@api.route('/disabled/<string:project>')
class Disabled(Resource):
    def get(self, project):
        DISABLED_FILE = os.path.join(TRAINING_IMG, project, 'disabled.txt')

        if not os.path.isfile(DISABLED_FILE):
            return jsonify([])

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
