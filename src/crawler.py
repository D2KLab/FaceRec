import argparse
import logging
import os
import sys
import time

import cv2
from icrawler.builtin import GoogleImageCrawler

from .FaceDetector import FaceDetector

logger = logging.getLogger('crawler')


def main(keyword, max_num=50, project='general'):
    if not keyword:
        raise ValueError('Keyword parameter is required.')

    keyword = keyword.strip()

    logger.info('[%s] Crawler run for: %s' % (project, keyword))
    print('[%s] Crawler run for: %s' % (project, keyword))

    image_dir = os.path.expanduser(os.path.join('data/training_img/', project, keyword.replace(" ", "_")))
    os.makedirs(image_dir, exist_ok=True)
    al_image_dir = os.path.expanduser(os.path.join('data/training_img_aligned/', project, keyword.replace(" ", "_")))
    os.makedirs(al_image_dir, exist_ok=True)

    google_crawler = GoogleImageCrawler(feeder_threads=10, parser_threads=10, log_level=logging.DEBUG,
                                        downloader_threads=25, storage={'root_dir': image_dir})
    # filters = dict(type='photo')  # I find photo more accurate than 'face'
    start = time.time()
    google_crawler.crawl(keyword=keyword, offset=0, max_num=max_num,
                         min_size=(200, 200), max_size=None, file_idx_offset=0)

    detector = FaceDetector(detect_multiple_faces=True)

    for f in sorted(os.listdir(image_dir)):
        filename = f.rsplit('.', 1)[0]
        print(filename)
        image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, f)), cv2.COLOR_BGR2GRAY)
        extracted_faces = detector.extract(image)
        for i, face in enumerate(extracted_faces):
            output_filename = os.path.join(al_image_dir, '%s_%d.png' % (filename, i))
            cv2.imwrite(output_filename, face)

    end = time.time()
    logger.info("Time elapsed: %.2f seconds", end - start)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', "--keyword", type=str, required=True,
                        help='Keyword for searching')
    parser.add_argument("--max_num", type=int, default=20,
                        help='Max number of images to download')
    parser.add_argument("--project", type=str, default='general',
                        help='Name of the collection to be part of')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.keyword, args.max_num, args.project)
