import os
import json
import datetime
from tinydb import TinyDB, Query
from flask import Flask, request, jsonify


from src import *

TRAINING_IMG = 'data/training_img'
os.makedirs('database', exist_ok=True)

app = Flask(__name__)
db = TinyDB('database/db.json')


@app.route('/crawler')
def crawl():
    q = request.args.get('q')
    if q is None:
        raise ValueError('Missing required parameter: q')
    for keyword in q.split(';'):
        crawler.main(keyword, max_num=50)
    return jsonify({
        'task': 'crawl',
        'time': str(datetime.datetime.now()),
        'status': 'ok'
    })


@app.route('/train')
def train():
    FaceDetector.main()
    classifier.main('TRAIN', classifier='SVM', batch_size=200)
    return jsonify({
        'task': 'train',
        'time': str(datetime.datetime.now()),
        'status': 'ok'
    })


@app.route('/recognise')
def recognise():
    args = request.args.to_dict()
    speedup = request.args.get('speedup', type=int, default=1)
    no_cache = 'no_cache' in args

    results = None
    if not no_cache:
        results = db.search(Query().status == 'ok')
        if results and len(results) > 0:
            results = results[0]

    if not results:
        r = FaceRecogniser.main(video_speedup=speedup, confidence_threshold=0.7)
        results = {
            'task': 'recognise',
            'status': 'ok',
            'time': str(datetime.datetime.now()),
            'results': r
        }

        db.insert(results)

    # with open('recognise.json', 'w') as outfile:
    #     json.dump(r, outfile)

    return jsonify(results)


@app.errorhandler(ValueError)
def handle_invalid_usage(error):
    response = jsonify({
        'status': 'error',
        'error': str(error),
        'time': str(datetime.datetime.now())
    })
    response.status_code = 422
    return response


if __name__ == '__main__':
    app.run()
