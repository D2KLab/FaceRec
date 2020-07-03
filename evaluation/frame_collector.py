import os
import re

import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN(min_face_size=25)


class FrameCollector:
    def __init__(self, video_path, project='general'):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)

        # setup all paths
        self.frames_path = './frames/%s/' % project
        os.makedirs(self.frames_path, exist_ok=True)

        # frames per second
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.scale_rate = 0.9 if width > 700 else 1

    def run(self, frame_no=None):
        if frame_no is None:
            return 0

        # iterate over the frames
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        # read the frame
        ret, frame = self.video_capture.retrieve()

        if frame is None:
            raise RuntimeError(
                'Frame %d not found in %s . Length: %d' % (frame_no, self.video_path, self.video_length))

        frame = cv2.resize(frame, (0, 0), fx=self.scale_rate, fy=self.scale_rate)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
        bounding_boxes = detector.detect_faces(rgb_frame)
        if len(bounding_boxes) == 0:
            return 0

        # if there is at least 1 face, try predict

        # save the frame
        fname = self.video_path.split('/')[-1].split('.')[0]
        filename = os.path.join(self.frames_path, fname + '_%d.jpg' % frame_no)
        cv2.imwrite(filename, frame)
        return filename
