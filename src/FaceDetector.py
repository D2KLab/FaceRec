import argparse
import os
import sys

import numpy as np
from PIL import Image
from mtcnn import MTCNN

import tensorflow as tf
from .utils import utils


def main(project='general', image_size=160, margin=44,
         detect_multiple_faces=False, discard_disabled=True):
    input_dir = os.path.join('data/training_img/', project)
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.join('data/training_img_aligned/', project)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data, labels, paths, _ = utils.load_dataset(input_dir, keep_original_size=True)

    detector = MTCNN()

    nrof_successfully_aligned = 0

    disabled = []
    if discard_disabled:
        with open(os.path.join(input_dir, 'disabled.txt')) as f:
            disabled = f.read().splitlines()

    for img, label, path in zip(data, labels, paths):
        if path in disabled:
            pass

        output_class_dir = os.path.join(output_dir, label.replace(' ', '_'))
        os.makedirs(output_class_dir, exist_ok=True)

        filename = os.path.splitext(os.path.split(path)[1])[0]
        output_filename = os.path.join(output_class_dir, filename + '.png')

        bounding_boxes = detector.detect_faces(img)
        nrof_faces = len(bounding_boxes)
        if nrof_faces > 0:
            det = np.array([utils.fix_box(b['box']) for b in bounding_boxes])
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = det[:, 2] * det[:, 3]
                    img_center = img_size / 2
                    offsets = np.vstack([det[:, 0] + (det[:, 2] / 2) - img_center[1],
                                         det[:, 1] + (det[:, 3] / 2) - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    # some extra weight on the centering
                    index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(utils.xywh2rect(*det))
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = np.array(Image.fromarray(cropped)
                                  .resize((image_size, image_size), resample=Image.BILINEAR)
                                  .convert('L'))
                filename_base, file_extension = os.path.splitext(output_filename)
                if detect_multiple_faces:
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                else:
                    output_filename_n = "{}{}".format(filename_base, file_extension)
                Image.fromarray(scaled).save(output_filename_n)
                nrof_successfully_aligned += 1
        else:
            print('Unable to align "%s"' % path)

    print('Total number of images: %d' % len(paths))
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('project', type=str, default='general',
                        help='Name of the collection to be part of')
    parser.add_argument('--image_size', type=int, default=160,
                        help='Image size (height, width) in pixels')
    parser.add_argument('--margin', type=int, default=44,
                        help='Margin for the crop around the bounding box (height, width) in pixels')
    parser.add_argument('--detect_multiple_faces', type=bool, default=False,
                        help='Detect and align multiple faces per image')
    parser.add_argument('--discard_disabled', type=bool, default=False,
                        help='If true, skip the images in the file "disabled.txt"')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.project, args.image_size, args.margin, args.detect_multiple_faces, args.discard_disabled)
