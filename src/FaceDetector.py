import argparse
import os
import sys

import cv2
import numpy as np
from mtcnn import MTCNN

from .FaceAligner import FaceAligner
from .utils import utils


class FaceDetector:
    def __init__(self, image_size=160, margin=10, detect_multiple_faces=False):
        self.aligner = FaceAligner(desiredFaceWidth=image_size, margin=margin)
        self.detector = MTCNN()
        self.detect_multiple_faces = detect_multiple_faces

    def extract(self, img):
        """ Extract the portions of a single img or frame including faces """
        bounding_box, landmarks = self.detect(img)
        return [self.aligner.align(img, det) for det in zip(bounding_box, landmarks)]

    def detect(self, img):
        bounding_boxes = self.detector.detect_faces(img)
        nrof_faces = len(bounding_boxes)
        if nrof_faces <= 0:
            return [], []

        det = np.array([utils.fix_box(b['box']) for b in bounding_boxes])
        img_size = np.asarray(img.shape)[0:2]

        if nrof_faces > 1 and not self.detect_multiple_faces:
            # select the biggest and most central
            bounding_box_size = det[:, 2] * det[:, 3]
            img_center = img_size / 2
            offsets = np.vstack([det[:, 0] + (det[:, 2] / 2) - img_center[1],
                                 det[:, 1] + (det[:, 3] / 2) - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # some extra weight on the centering
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det_arr = [det[index, :]]
            landmarks = [bounding_boxes[index]['keypoints']]
        else:
            det_arr = [np.squeeze(d) for d in det]
            landmarks = [b['keypoints'] for b in bounding_boxes]

        return det_arr, landmarks


def main(project='general', image_size=160, margin=10, detect_multiple_faces=False):
    input_dir = os.path.join('data/training_img/', project)
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.join('data/training_img_aligned/', project)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data, labels, paths, _ = utils.load_dataset(input_dir, keep_original_size=True)

    detector = FaceDetector(image_size, margin, detect_multiple_faces)

    nrof_successfully_aligned = 0

    for img, label, path in zip(data, labels, paths):
        output_class_dir = os.path.join(output_dir, label.replace(' ', '_'))
        os.makedirs(output_class_dir, exist_ok=True)

        filename = os.path.splitext(os.path.split(path)[1])[0]

        extracted_faces = detector.extract(img)
        if len(extracted_faces) == 0:
            print('Unable to detect faces in %s' % path)
            continue

        nrof_successfully_aligned += 1

        # save images to file
        for i, face in enumerate(extracted_faces):
            suffix = ('_%d' % i) if detect_multiple_faces else ''
            output_filename = os.path.join(output_class_dir, filename + suffix + '.png')
            cv2.imwrite(output_filename, face)

    print('Total number of images: %d' % len(paths))
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('project', type=str, default='general',
                        help='Name of the collection to be part of')
    parser.add_argument('--image_size', type=int, default=160,
                        help='Image size (height, width) in pixels')
    parser.add_argument('--margin', type=int, default=10,
                        help='Margin for the crop around the bounding box (height, width) in pixels')
    parser.add_argument('--detect_multiple_faces', default=False, action='store_true',
                        help='Detect and align multiple faces per image')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.project, args.image_size, args.margin, args.detect_multiple_faces)
