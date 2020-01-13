import argparse
import os
import pickle

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from .utils import facenet, utils
from .align import detect_face

ALIGN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align")


def main(video_path, output_path='data/cluster.txt', facenet_model_path='model/20180402-114759.pb',
         classifier_path='classifier/classifier.pkl', video_speedup=50, folder_containing_frame=None,
         confidence_threshold=0.6):
    video_capture = cv2.VideoCapture(video_path)

    if folder_containing_frame is None:
        folder_containing_frame = utils.generate_output_path('./data/frames', video_path)

    minsize = 20  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    input_image_size = 160

    # Load classifier
    classifier_filename = os.path.expanduser(classifier_path)
    with open(classifier_filename, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file: %s" % classifier_filename)

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
            video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            matches = []

            # iterate over the frames
            for frame_no in np.arange(0, video_length, video_speedup):
                print(frame_no)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

                # read the frame
                ret, frame = video_capture.retrieve()

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

                faces_found = bounding_boxes.shape[0]
                if faces_found == 0:
                    continue

                det = np.array(bounding_boxes[:, 0:4], dtype=np.int32)
                for bb in det:

                    # inner exception
                    if bb[0] <= 0 or bb[1] <= 0 or bb[2] >= len(frame[0]) or bb[3] >= len(frame):
                        print('face is inner of range!')
                        continue

                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = cv2.resize(cropped, (input_image_size, input_image_size),
                                        interpolation=cv2.INTER_CUBIC)

                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                    # convert to array and predict among the known ones
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
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        text_x = bb[0]
                        text_y = bb[3] + 20
                        cv2.putText(frame, class_names[best_class_indices[0]], (text_x, text_y),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)

                        matches.append({
                            'name': best_name,
                            'confidence': best_class_probabilities,
                            'video': video_path,
                            'frame': frame_no,
                            'npt': utils.frame2npt(frame_no, fps),
                            'bounding': utils.rect2xywh(bb[0], bb[1], bb[2], bb[3])
                        })

                        with open(output_path, 'a+') as f:
                            f.write(str(frame_no) + ',' + class_names[best_class_indices[0]] + "\n")
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
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--classifier_path', type=str,
                        help='Path to KNN classifier',
                        default="classifier/classifier.pkl")
    parser.add_argument('--video_speedup', type=int,
                        help='Speed up for the video', default=50)
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
    main(video, args.output_path, args.model_path, args.classifier_path,
         args.video_speedup, args.folder_containing_frame, args.confidence_threshold)
