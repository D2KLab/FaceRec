import argparse
import os
import pickle
import csv
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from .align import detect_face
from .utils import facenet, utils
from .SORT.sort import Sort
from .utils.face_utils import judge_side_face

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")

colours = np.random.rand(32, 3)

file_to_be_close = []


def export_frame(input_frame, d, classname, frame_num, frames_path, writer):
    frame = input_frame.copy()
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
    cv2.putText(frame, 'a' + classname, (d[0] - 10, d[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colours[d[4] % 32, :] * 255, 2)

    print([str(i) for i in d] + [classname, str(frame_num)])
    writer.writerow([str(i) for i in d] + [classname, str(frame_num)])

    filename = 'frame' + str(frame_num) + '.jpg'
    cv2.imwrite(os.path.join(frames_path, filename), frame)


def init_csv(path, fieldnames):
    file = open(path, 'w')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)
    file_to_be_close.append(file)
    return writer


def main(video_path, face_fragment_path, frames_path,
         trackers_csv='data/trackers.csv',
         predictions_csv='data/predictions.csv',
         trackout_csv='data/trackout.csv',
         classifier_path='classifier/classifier.pkl',
         facenet_model_path='model/20180402-114759.pb', video_speedup=1, export_frames=False):
    video_capture = utils.get_capture(video_path)

    if face_fragment_path is None:
        face_fragment_path = utils.generate_output_path('./data/cluster', video_path)

    if export_frames:
        if frames_path is None:
            frames_path = utils.generate_output_path('./data/frames', video_path)

    minsize = 50  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    input_image_size = 160
    scale_rate = 0.9  # if set it smaller will make input frames smaller

    # Load classifier
    classifier_filename = os.path.expanduser(classifier_path)
    with open(classifier_filename, 'rb') as f:
        (classifier, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename)

    # Get the path of the facenet model and load it
    facenet.load_model(facenet_model_path)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    # init csv outputs
    trackers_writer = init_csv(trackers_csv, ['x1', 'y1', 'x2', 'y2', 'track_id', 'frame'])
    predictions_writer = init_csv(predictions_csv, ['name', 'track_id'])
    trackout_writer = init_csv(trackout_csv, ['x1', 'y1', 'x2', 'y2', 'track_id', 'name', 'frame'])

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Bounding box
            pnet, rnet, onet = detect_face.create_mtcnn(sess, ALIGN_MODEL_PATH)
            # Get the path of the facenet model and load it
            facenet.load_model(facenet_model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # frames per second
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames_passed = -1
            matches = []

            # start reading frame by frame
            while video_capture.grab():  # move pointer to next frame
                total_frames_passed += 1
                # Skip frames if video is to be speed up
                if video_speedup > 1 and total_frames_passed % video_speedup != 0:
                    continue

                # Otherwise read the frame
                ret, frame = video_capture.retrieve()

                face_list = []
                attribute_list = []
                frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                frame_height, frame_width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_size = np.asarray(frame.shape)[0:2]
                bounding_boxes, points = detect_face.detect_face(rgb_frame, minsize,
                                                                 pnet, rnet, onet, threshold, factor)
                # points are the face landmarks

                for i, item in enumerate(bounding_boxes):
                    f = round(item[4], 6)
                    if f <= 0.99:
                        continue
                    det = np.squeeze(item[0:4])
                    face_list.append(item)

                    # face cropped
                    bb = np.array(det, dtype=np.int32)
                    cropped = frame.copy()[bb[1]:bb[3], bb[0]:bb[2], :]

                    # use 5 face landmarks to judge the face is front or side
                    plist = np.squeeze(points[:, i]).tolist()
                    facial_landmarks = [[plist[j], plist[(j + 5)]] for j in range(5)]

                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                        np.array(facial_landmarks))

                    # face additional attribute
                    # (index 0:face score; index 1:0 represents front face and 1 for side face )
                    attribute_list.append([cropped, item[4], dist_rate, high_ratio_variance, width_rate])

                final_faces = np.array(face_list)
                trackers = tracker.update(final_faces, img_size, face_fragment_path, attribute_list, rgb_frame)

                for d in trackers:
                    d = d.astype(np.int32)
                    # print(d)
                    print(total_frames_passed)
                    # FIXME how this is possible?
                    if any(i < 0 for i in d) \
                            or d[0] >= frame_width or d[2] >= frame_width \
                            or d[1] >= frame_height or d[3] >= frame_height:
                        print('Error tracker: ')
                        print(d)
                        continue

                    trackers_writer.writerow([str(i) for i in d] + [str(total_frames_passed)])

                    # cutting the img on the face
                    trackers_cropped = frame[d[1]:d[3], d[0]:d[2], :]
                    scaled = cv2.resize(trackers_cropped, (input_image_size, input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)

                    # convert to array and predict among the known ones
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = classifier.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[best_class_indices[0]]

                    if best_class_probabilities > 0.09:  # TODO too low ?
                        predictions_writer.writerow([best_name, str(d[4])])
                        matches.append({
                            'name': best_name,
                            'track_id': d[4],
                            'video': video_path,
                            'frame': total_frames_passed,
                            'confidence': best_class_probabilities,
                            'npt': utils.frame2npt(total_frames_passed, fps),
                            'bounding': utils.rect2xywh(d[0], d[1], d[2], d[3])
                        })

                        if export_frames:
                            export_frame(frame, d, best_name, total_frames_passed, frames_path, trackout_writer)

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
    parser.add_argument('--model_path', type=str, default='model/20180402-114759.pb',
                        help='Path to the facenet embedding model')

    # paths for the output
    parser.add_argument('--face_fragment_path', type=str,
                        help='Path for saving the fragment of image containing the detected face.\n'
                             'By default is in `data\\cluster\\<video_name>`')
    parser.add_argument('--frames_path', type=str,
                        help='If `--export_frames` is set, path for saving the annotated frames.\n'
                             'By default is in `data\\frames\\<video_name>`')
    parser.add_argument('--trackers_csv', type=str, default='data/trackers.csv',
                        help='Path to the csv file for writing the computed trackers')
    parser.add_argument('--predictions_csv', type=str, default='data/predictions.csv',
                        help='Path to the csv output of the frame-based predictions')
    parser.add_argument('--trackout_csv', type=str, default='data/trackout.csv',
                        help='Path to the csv output of the frame-based predictions')

    # parameters
    parser.add_argument('--video_speedup', type=int, default=50,
                        help='Speed up for the video')
    parser.add_argument('--export_frames', default=False, action='store_true',
                        help='If specified, export the annotated frames')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.video, args.face_fragment_path, args.frames_path,
         args.trackers_csv, args.predictions_csv, args.trackout_csv,
         args.classifier_path, args.model_path,
         args.video_speedup, args.export_frames)
