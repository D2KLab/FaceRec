import argparse
import collections
import os
import pickle

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from .utils import facenet
from .align import detect_face


def rect2xywh(x, y, x2, y2):
    w = x2 - x  # width
    h = y2 - y  # height

    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}


def frame2npt(frame, fps):
    return frame / fps


def main(video_path, output_path='data/cluster', model_path='model/20180402-114759.pb',
         classifier_path='classifier/classifier.pkl', video_speedup=50, folder_containing_frame='./data/frames',
         confidence_threshold=0.795):
    video_name = os.path.join('video', video_path.replace('/', '_') + '.mp4')
    if not os.path.isfile(video_name):
        video_name = video_path
    if not os.path.isfile(video_name):  # still
        raise FileNotFoundError('video not found: %s' % video_name)

    os.makedirs(folder_containing_frame, exist_ok=True)

    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    input_image_size = 160

    # Get the path of the classifier and load it
    project_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_filename_exp = os.path.expanduser(classifier_path)

    with open(classifier_filename_exp, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename_exp)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Bounding box
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
            # Get the path of the facenet model and load it
            facenet.load_model(model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Start video capture
            person_detected = collections.Counter()

            # for filename in os.listdir(video_dir):
            #     suffix = filename.split('.')[-1]
            #     if suffix != 'mp4' and suffix != 'avi':
            #         continue
            #     video_name = os.path.join(video_dir, filename)
            print(video_name)
            video_capture = cv2.VideoCapture(video_name)

            # frames per second
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            total_frames_passed = -1

            matches = []
            while video_capture.grab():  # move pointer to next frame
                total_frames_passed += 1

                # Skip frames if video is to be speed up
                if video_speedup > 1:
                    if total_frames_passed % video_speedup != 0:
                        continue

                # Otherwise read the frame
                ret, frame = video_capture.retrieve()

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faces_found = bounding_boxes.shape[0]

                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]

                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (input_image_size, input_image_size),
                                            interpolation=cv2.INTER_CUBIC)

                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices][
                            0]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Confidence: {}".format(best_name, best_class_probabilities))
                        if best_class_probabilities > confidence_threshold:
                            # boxing face
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            cv2.putText(frame, class_names[best_class_indices[0]], (text_x, text_y),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1

                            matches.append({
                                'name': best_name,
                                'confidence': best_class_probabilities,
                                'video': video_path,
                                'frame': total_frames_passed,
                                'npt': frame2npt(total_frames_passed, fps),
                                'bounding': rect2xywh(bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                            })

                            with open(output_path, 'a+') as f:
                                f.write(str(total_frames_passed) + ',' + class_names[best_class_indices[0]] + "\n")
                            frame_number = 'frame' + str(total_frames_passed) + '.jpg'
                            filename = os.path.join(folder_containing_frame, frame_number)
                            cv2.imwrite(filename, frame)

    cv2.destroyAllWindows()
    return matches


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help='Path to the video to be converted.')
    parser.add_argument('--output_path', type=str,
                        help='Path to the txt output file',
                        default='data/cluster')
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--classifer_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier.pkl")
    parser.add_argument('--video_speedup', type=int,
                        help='Speed up for the video', default=50)
    parser.add_argument("--folder_containing_frame", type=str,
                        help='Path to the out data directory containing frames.',
                        default="./data/frames")
    parser.add_argument("--confidence_threshold", type=str,
                        help='Confidence threshold for having a positive face match.',
                        default=0.795)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.video, args.output_path, args.model_path, args.classifier_path,
         args.video_speedup, args.folder_containing_frame, args.confidence_threshold)
