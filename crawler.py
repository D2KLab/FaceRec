import argparse
import sys
import time
import os
from icrawler.builtin import GoogleImageCrawler
import face_recognition

def main(args):
    keyword = args.keyword
    max_num = args.max_num
    image_dir = os.path.expanduser(args.image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    google_crawler = GoogleImageCrawler(feeder_threads=10, parser_threads=10,
                                        downloader_threads=25, storage={'root_dir': image_dir})
    filters = dict(type = 'face')
    start = time.time()    
    google_crawler.crawl(keyword=keyword, filters=filters, offset=0, max_num= max_num,
                     min_size=(200,200), max_size=None, file_idx_offset=0)
    link_names = []
    for (rootDir, dirNames, filenames) in os.walk(image_dir):
        for filename in filenames:
            link_names.append(os.path.join(rootDir, filename))
    for filename in link_names:
        image = face_recognition.load_image_file(filename)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) >=2:
            os.remove(filename)                    
    end = time.time()
    print("Time elapsed:", end - start)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str,
                        help='keyword for searching.')
    parser.add_argument("--max_num", type=int,
                        help='the maximum number of images downloaded.')    
    parser.add_argument("--image_dir", type=str, help='Directory with downloaded images.')     
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
