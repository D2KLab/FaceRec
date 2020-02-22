import os
import sys
import time
import argparse
import logging
import face_recognition
from icrawler.builtin import GoogleImageCrawler

logger = logging.getLogger('crawler')


def main(keyword, max_num=20, image_dir=None):
    if not keyword:
        raise ValueError('Keyword parameter is required.')

    keyword = keyword.strip()

    logger.info('Crawler run for: %s' % keyword)
    print('Crawler run for: %s' % keyword)

    if image_dir is None:
        image_dir = 'data/training_img/%s' % keyword.replace(" ", "_")

    image_dir = os.path.expanduser(image_dir)
    os.makedirs(image_dir, exist_ok=True)

    google_crawler = GoogleImageCrawler(feeder_threads=10, parser_threads=10, log_level=logging.DEBUG,
                                        downloader_threads=25, storage={'root_dir': image_dir})
    # filters = dict(type='photo')  # I find photo more accurate than 'face'
    start = time.time()
    google_crawler.crawl(keyword=keyword, offset=0, max_num=max_num,
                         min_size=(200, 200), max_size=None, file_idx_offset=0)

    for (rootDir, dirNames, filenames) in os.walk(image_dir):
        for f in filenames:
            filename = os.path.join(rootDir, f)
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            # I just keep images with 1 single face
            if len(face_locations) != 1:
                os.remove(filename)
    end = time.time()
    logger.info("Time elapsed: %.2f seconds", end - start)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', "--keyword", type=str, required=True,
                        help='Keyword for searching')
    parser.add_argument("--max_num", type=int, default=20,
                        help='Max number of images to download')
    parser.add_argument("--image_dir", type=str, default=None,
                        help='Output folder for downloaded images')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.keyword, args.max_num, args.image_dir)
