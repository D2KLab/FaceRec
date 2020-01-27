import os
import time
import datetime
from tinydb import TinyDB, Query, where
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

import semantifier
from src import *
from src.utils import utils

VIDEO_DIR = os.path.join(os.getcwd(), 'video')

TRAINING_IMG = 'data/training_img'
os.makedirs('database', exist_ok=True)

app = Flask(__name__)
CORS(app)
db_detection = TinyDB('database/detection.json')
db_tracking = TinyDB('database/tracking.json')


def now():
    return datetime.datetime.now().isoformat()


@app.route('/crawler')
def crawl():
    start_time = time.time()

    q = request.args.get('q')
    if q is None:
        raise ValueError('Missing required parameter: q')
    for keyword in q.split(';'):
        crawler.main(keyword, max_num=50)
    return jsonify({
        'task': 'crawl',
        'time': now(),
        'execution_time': (time.time() - start_time),
        'status': 'ok'
    })


@app.route('/train')
def train():
    start_time = time.time()

    FaceDetector.main()
    classifier.main('TRAIN', classifier='SVM', batch_size=200)
    return jsonify({
        'task': 'train',
        'time': now(),
        'execution_time': (time.time() - start_time),
        'status': 'ok'
    })


# http://127.0.0.1:5000/track?speedup=25&video=video/yle_a-studio_8a3a9588e0f58e1e40bfd30198274cb0ce27984e.mp4
# http://127.0.0.1:5000/track?format=ttl&video=http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
@app.route('/track')
def track():
    start_time = time.time()

    video = request.args.get('video')
    speedup = request.args.get('speedup', type=int, default=25)
    no_cache = 'no_cache' in request.args.to_dict()

    results = None
    info = None
    if not no_cache:
        results = db_tracking.search(Query().video == video)
        if results and len(results) > 0:
            results = results[0]

    if not results:
        video_path = video
        if video.startswith('http'):  # it is a uri!
            video_path, info = utils.uri2video(video)
        elif not os.path.isfile(video):
            raise FileNotFoundError('video not found: %s' % video)

        r = tracker.main(video_path, video_speedup=speedup)
        results = {
            'task': 'tracking',
            'status': 'ok',
            'execution_time': (time.time() - start_time),
            'time': now(),
            'video': video,
            'info': info,
            'results': r
        }

        # TODO insert aliases in the cache
        # delete previous results
        db_tracking.remove(where('video') == video)
        db_tracking.insert(results)

    clusters = clusterize.main(clusterize.from_dict(results['results']), confidence_threshold=0.5, merge_cluster=True)
    results = {
        'task': 'recognise',
        'status': 'ok',
        'execution_time': (time.time() - start_time),
        'time': now(),
        'video': video,
        'info': info,
        'results': clusters
    }

    fmt = request.args.get('format')
    if fmt == 'ttl':
        return Response(semantifier.semantify(results), mimetype='text/turtle')

    return jsonify(results)


# http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
# http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/eurovaalit-2019-kuka-johtaa-eurooppaa/0460c1b7d735e3fc796aa2829811aa1ae5dc9fa8
# http://127.0.0.1:5000/recognise?speedup=50&format=ttl&video=yle/eurovaalit-2019-kuka-johtaa-eurooppaa/d9d05488b35db559cdef35bac95f518ee0dda76a
# http://127.0.0.1:5000/recognise?speedup=50&format=ttl&no_cache&video=http://data.memad.eu/yle/a-studio/8a3a9588e0f58e1e40bfd30198274cb0ce27984e
@app.route('/recognise')
def recognise():
    start_time = time.time()

    video = request.args.get('video')
    speedup = request.args.get('speedup', type=int, default=25)
    no_cache = 'no_cache' in request.args.to_dict()

    results = None
    info = None
    if not no_cache:
        results = db_detection.search(Query().video == video)
        if results and len(results) > 0:
            results = results[0]

    if not results:
        video_path = video
        if video.startswith('http'):  # it is a uri!
            video_path, info = utils.uri2video(video)
        elif not os.path.isfile(video):
            raise FileNotFoundError('video not found: %s' % video)

        r = FaceRecogniser.main(video_path, video_speedup=speedup, confidence_threshold=0.2)
        results = {
            'task': 'recognise',
            'status': 'ok',
            'execution_time': (time.time() - start_time),
            'time': now(),
            'video': video,
            'info': info,
            'results': r
        }

        # TODO insert aliases in the cache
        # delete previous results
        db_detection.remove(where('video') == video)
        db_detection.insert(results)

    # with open('recognise.json', 'w') as outfile:
    #     json.dump(r, outfile)
    fmt = request.args.get('format')
    if fmt == 'ttl':
        return Response(semantifier.semantify(results), mimetype='text/turtle')

    return jsonify(results)


@app.route('/video/<path:path>')
def send_video(path):
    return send_from_directory(VIDEO_DIR, path, as_attachment=True)


@app.errorhandler(ValueError)
def handle_invalid_usage(error):
    response = jsonify({
        'status': 'error',
        'error': str(error),
        'time': now()
    })
    response.status_code = 422
    return response


if __name__ == '__main__':
    app.run()
