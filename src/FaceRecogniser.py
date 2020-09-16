import argparse
import os
import pickle

import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import scipy.cluster as cluster

from .utils import utils


def select_best(predictions, class_names):
    best_index = np.argmax(predictions)
    best_prob = predictions[best_index]
    best_name = class_names[best_index]
    return best_name, best_prob


class Classifier:
    def __init__(self,
                 classifier_path='data/classifier/classifier.pkl',
                 facenet_model='./model/facenet_keras.h5',
                 facenet_weights='./model/facenet_keras_weights.h5'):
        self.image_size = 160

        self.facenet = load_model(facenet_model, compile=False)
        self.facenet.load_weights(facenet_weights)
        self.features = []
        self.meta = []
        self.collect_features = False

        # Load classifier
        classifier_filename = os.path.expanduser(classifier_path)
        with open(classifier_filename, 'rb') as f:
            (classifier, class_names) = pickle.load(f)
            print("Loaded classifier file: %s" % classifier_filename)
            self.classifier = classifier
            self.class_names = class_names

    def predict(self, img, meta=None):
        scaledx = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        scaled = scaledx.reshape(-1, self.image_size, self.image_size, 3)
        # convert to array and predict among the known ones
        emb_arrayx = [utils.get_embedding(self.facenet, face_pixels) for face_pixels in scaled]
        emb_array = np.asarray(emb_arrayx)
        if self.collect_features:
            _, _, rect, _ = meta
            dim = (rect[3] - rect[1]) * (rect[2] - rect[0])
            if dim >= 2000:
                self.features.append(emb_array[0])
                self.meta.append(meta)
        return self.classifier.predict_proba(emb_array).flatten()

    def predict_best(self, img, meta=None):
        predictions = self.predict(img, meta)
        return select_best(predictions, self.class_names)

    def cluster_features(self, clustering_distance=14, distance_threshold=1.3, side_face_threshold=0.6, min_samples=7,
                         max_samples=5, min_involved_tracks=3):
        # print('NOW CLUSTERING', len(self.features),
        #       'feature vectors of dimensionality', len(self.features[0]),
        #       'to', nclusters, 'clusters and showing max', maxsamples,
        #       'samples of clusters that contain min', minsamples, 'vectors')
        features = np.array(self.features)
        link = cluster.hierarchy.linkage(features, method='complete')
        fc = cluster.hierarchy.fcluster(link, clustering_distance, criterion='distance')
        clusters = []
        for i in np.unique(fc):
            cur_indexes = [j for j, k in enumerate(fc) if k == i]
            x = np.array(features)[cur_indexes]
            y = np.array(self.meta)[cur_indexes]

            m = np.mean(x, axis=0)
            avg = np.linalg.norm(x - m, axis=1)
            e = sorted(zip(y, avg), key=lambda a: a[1])

            if len(e) < min_samples:
                continue

            avg = np.mean(avg)
            distance_score = avg / len(e)

            if distance_score > distance_threshold:
                continue

            elements = [x[0] for x in e]
            involved_tracks = np.unique([track for frame_no, track, bb, ld in elements])
            if len(involved_tracks) < min_involved_tracks:
                continue
            # TODO
            # if any([x in official_cluster_mapping for x in involved_tracks]):
            #     continue

            lds = [ld for frame_no, track, bb, ld in elements[0:5]]
            side_face_score = np.sum([(1 / (i + 1) * ld) for i, ld in enumerate(lds)]) / len(lds)
            if side_face_score > side_face_threshold:
                continue

            clusters.append({
                'id': len(clusters),
                'elements': [{
                    'frame': int(frame_no),
                    'track': int(track),
                    'rect': [int(b) for b in bb]
                } for frame_no, track, bb, ld in elements[0:max_samples]]
            })
        return clusters


def main(video_path, output_path='data/cluster.txt',
         classifier_path='classifier/classifier.pkl', video_speedup=25, folder_containing_frame=None,
         confidence_threshold=0.6):
    video_capture = cv2.VideoCapture(video_path)

    if folder_containing_frame is None:
        folder_containing_frame = utils.generate_output_path('./data/frames', '', video_path=video_path)

    detector = MTCNN()

    # Load classifier
    classifier = Classifier(classifier_path)

    # frames per second
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    matches = []

    # iterate over the frames
    for frame_no in np.arange(0, video_length, video_speedup):
        print('frame %d/%d' % (frame_no, video_length))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        # read the frame
        ret, frame = video_capture.retrieve()

        bounding_boxes = detector.detect_faces(frame)
        if len(bounding_boxes) == 0:
            continue

        for item in bounding_boxes:
            bb = utils.xywh2rect(*utils.fix_box(item['box']))

            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            best_name, best_prob = classifier.predict_best(cropped)

            if best_prob > confidence_threshold:
                # boxing face
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                text_x = bb[0]
                text_y = bb[3] + 20
                cv2.putText(frame, best_name, (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 255), thickness=1, lineType=2)

                matches.append({
                    'name': best_name,
                    'confidence': best_prob,
                    'video': video_path,
                    'start_frame': frame_no,
                    'start_npt': utils.frame2npt(frame_no, fps),
                    'bounding': utils.rect2xywh(*bb),
                    'rect': bb
                })

                with open(output_path, 'a+') as f:
                    f.write(str(frame_no) + ',' + best_name + "\n")
                frame_number = 'frame' + str(frame_no) + '.jpg'
                filename = os.path.join(folder_containing_frame, frame_number)
                cv2.imwrite(filename, frame)

    return matches


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument('--output_path', type=str,
                        help='Path to the txt output file',
                        default='data/cluster.txt')
    parser.add_argument('--classifier_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier.pkl")
    parser.add_argument('--video_speedup', type=int,
                        help='Speed up for the video', default=25)
    parser.add_argument("--folder_containing_frame", type=str,
                        help='Path to the out data directory containing frames.',
                        default=None)
    parser.add_argument("--confidence_threshold", type=float,
                        help='Confidence threshold for having a positive face match.',
                        default=0.795)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    video = utils.normalize_video(args.video)
    main(video, args.output_path, args.classifier_path,
         args.video_speedup, args.folder_containing_frame, args.confidence_threshold)
