import argparse
import os
import pickle

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from .align import detect_face
from .utils import facenet, utils
from .SORT.sort import Sort
from .utils.face_utils import judge_side_face

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")

colours = np.random.rand(32, 3)


def export_frame(input_frame, d, dict_obid_classname, total_frames_passed, folder_containing_frame, output):
    frame = input_frame.copy()
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
    cv2.putText(frame, 'a' + dict_obid_classname, (d[0] - 10, d[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colours[d[4] % 32, :] * 255, 2)

    with open(output, 'a+') as f:
        f.write(" ".join(map(str, d)) + '.' + dict_obid_classname + '.' + str(
            total_frames_passed) + "\n")
    frame_number = 'frame' + str(total_frames_passed) + '.jpg'
    name = os.path.join(folder_containing_frame, frame_number)
    print(name)
    cv2.imwrite(name, frame)


def main(video_path, output_path, all_trackers_saved='data/all_trackers_saved.txt',
         obid_mapping_classnames='data/obid_mapping_classnames.txt',
         classifier_path='classifier/classifier.pkl',
         facenet_model_path='model/20180402-114759.pb', video_speedup=1, export_frames=False):
    video_capture = utils.get_capture(video_path)

    if output_path is None:
        output_path = utils.generate_output_path('./data/cluster', video_path)

    minsize = 50  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    input_image_size = 160
    scale_rate = 0.9  # if set it smaller will make input frames smaller

    # Load classifier
    classifier_filename = os.path.expanduser(classifier_path)
    with open(classifier_filename, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

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
                additional_attribute_list = []
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
                    additional_attribute_list.append([cropped, item[4], dist_rate, high_ratio_variance, width_rate])

                final_faces = np.array(face_list)
                trackers = tracker.update(final_faces, img_size, output_path, additional_attribute_list, rgb_frame)

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
                        with open(obid_mapping_classnames, 'a+') as f:
                            f.write('error tracker.%d\n' % total_frames_passed)
                        continue

                    with open(all_trackers_saved, 'a+') as f:
                        f.write(" ".join(map(str, d)) + '.%d\n' % total_frames_passed)

                    # cutting the img on the face
                    trackers_cropped = frame[d[1]:d[3], d[0]:d[2], :]
                    scaled = cv2.resize(trackers_cropped, (input_image_size, input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)

                    # convert to array and predict among the known ones
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                    best_name = class_names[best_class_indices[0]]

                    if best_class_probabilities > 0.09:  # TODO too low ?
                        with open(obid_mapping_classnames, 'a+') as f:
                            f.write(best_name + '.' + str(d[4]) + "\n")
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
                            os.makedirs('data/frames', exist_ok=True)
                            export_frame(frame, d, str(d[4]), total_frames_passed,
                                         'data/frames', 'data/track_out.txt')

            return matches


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=True,
                        help='Path or URI of the video to be analysed.')
    parser.add_argument('--output_path', type=str,
                        help='Path to the cluster folder')
    parser.add_argument('--all_trackers_saved', type=str, default='data/all_trackers_saved.txt',
                        help='Path to the txt file for all trackers saved')
    parser.add_argument('--obid_mapping_classnames', type=str, default='data/obid_mapping_classnames.txt',
                        help='Path to the txt output file for mapping file')
    parser.add_argument('--classifier_path', type=str, default='classifier/classifier.pkl',
                        help='Path to KNN classifier')
    parser.add_argument('--model_path', type=str, default='model/20180402-114759.pb',
                        help='Path to embedding model')
    parser.add_argument('--video_speedup', type=int,
                        help='Speed up for the video', default=50)
    parser.add_argument('--export_frames', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.video, args.output_path, args.all_trackers_saved,
         args.obid_mapping_classnames, args.classifier_path, args.model_path, args.video_speedup, args.export_frames)
