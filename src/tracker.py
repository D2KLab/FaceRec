import argparse
import csv
import os

import cv2
import numpy as np
from mtcnn import MTCNN

import src.database as database
from .FaceRecogniser import Classifier
from .SORT.sort import Sort
from .utils import utils
from .utils.face_utils import judge_side_face

colours = np.random.rand(32, 3)

file_to_be_close = []


def export_frame(input_frame, d, classname, frame_num, frames_path):
    frame = input_frame.copy()
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
    cv2.putText(frame, 'a' + classname, (d[0] - 10, d[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colours[d[4] % 32, :] * 255, 2)

    # print([str(i) for i in d] + [classname, str(frame_num)])

    filename = 'frame' + str(frame_num) + '.jpg'
    cv2.imwrite(os.path.join(frames_path, filename), frame)


def init_csv(path, fieldnames):
    file = open(path, 'w')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)
    file_to_be_close.append(file)
    return writer


def main(video_path, output_path=None,
         classifier_path='data/classifier/classifier.pkl',
         video_speedup=25, export_frames=False):
    print(video_path)

    video_capture = cv2.VideoCapture(video_path)

    if output_path is None:
        output_path = utils.generate_output_path('./data/out', video_path)

    # setup all paths
    cluster_path = os.path.join(output_path, 'cluster')
    frames_path = os.path.join(output_path, 'frames')
    if export_frames:
        os.makedirs(frames_path, exist_ok=True)
    trackers_csv = os.path.join(output_path, 'trackers.csv')
    predictions_csv = os.path.join(output_path, 'predictions.csv')

    # init csv outputs
    trackers_writer = init_csv(trackers_csv, ['x1', 'y1', 'x2', 'y2', 'track_id', 'frame'])
    predictions_writer = init_csv(predictions_csv, ['x1', 'y1', 'x2', 'y2',
                                                    'track_id', 'name', 'confidence', 'frame', 'tracker_sample', 'npt'])

    scale_rate = 0.9  # if set it smaller will make input frames smaller

    classifier = Classifier(classifier_path)

    detector = MTCNN(min_face_size=25)

    # init tracker
    tracker = Sort(min_hits=1)

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

        if frame is None:
            raise RuntimeError

        face_list = []
        attribute_list = []
        frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
        img_size = np.asarray(frame.shape)[0:2]
        bounding_boxes = detector.detect_faces(rgb_frame)
        # points are the face landmarks

        for item in bounding_boxes:
            bb = utils.xywh2rect(*utils.fix_box(item['box']))
            # use 5 face landmarks to judge the face is front or side
            facial_landmarks = list(item['keypoints'].values())

            f = round(item['confidence'], 6)
            if f <= 0.99:
                continue
            face_list.append(bb)

            # face cropped
            cropped = frame.copy()[bb[1]:bb[3], bb[0]:bb[2], :]

            dist_rate, high_ratio_variance, width_rate = judge_side_face(facial_landmarks)

            # face additional attribute
            # (index 0:face score; index 1:0 represents front face and 1 for side face )
            attribute_list.append([cropped, item['confidence'], dist_rate, high_ratio_variance, width_rate])

        trackers = tracker.update(np.array(face_list), img_size, cluster_path, attribute_list, rgb_frame)

        tracker_sample = tracker.frame_count
        # this is a counter of the frame analysed by the tracker (so normalised respect to the video_speedup)

        for d in trackers:
            d = d.astype(int)

            # FIXME how this is possible?
            if any(i < 0 for i in d) \
                    or d[0] >= frame_width or d[2] >= frame_width \
                    or d[1] >= frame_height or d[3] >= frame_height:
                print('Error tracker %d at frame %d:' % (d[4], frame_no))
                continue

            trackers_writer.writerow([str(i) for i in d] + [str(frame_no)])

            # cutting the img on the face
            trackers_cropped = frame[d[1]:d[3], d[0]:d[2], :]
            best_name, best_prob = classifier.predict_best(trackers_cropped)

            npt = utils.frame2npt(frame_no, fps)
            predictions_writer.writerow(
                [str(i) for i in d] + [best_name, best_prob, str(frame_no), tracker_sample, npt])

            match = {
                'name': best_name,
                'track_id': int(d[4]),
                'frame': int(frame_no),
                'confidence': best_prob,
                'tracker_sample': tracker_sample,
                'npt': npt,
                'locator': utils.clean_locator(video_path),
                'bounding': utils.rect2xywh(*d[0:4]),
                'rect': d[0:4].tolist()
            }
            matches.append(match)
            if database.is_on():
                database.insert_partial_analysis(match)

            if export_frames:
                export_frame(frame, d, best_name, frame_no, frames_path)

    # TODO final track

    if database.is_on():
        database.save_status(utils.clean_locator(video_path), 'COMPLETE')

    for f in file_to_be_close:
        f.close()
    return matches


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    # files required in input
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument('--classifier_path', type=str, default='classifier/classifier.pkl',
                        help='Path to the KNN classifier')

    # paths for the output
    parser.add_argument('--output', type=str,
                        help='Path for saving all the ouput of the script.\n'
                             'By default is in `data\\out\\<video_name>`')

    # parameters
    parser.add_argument('--video_speedup', type=int, default=25,
                        help='Speed up for the video')
    parser.add_argument('--export_frames', default=False, action='store_true',
                        help='If specified, export the annotated frames')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    video = utils.normalize_video(args.video)

    main(video, args.output,
         args.classifier_path,
         args.video_speedup, args.export_frames)
