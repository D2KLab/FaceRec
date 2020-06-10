import os
import re

import numpy as np
import cv2
from PIL import Image


def rect2xywh(x, y, x2, y2):
    w = x2 - x  # width
    h = y2 - y  # height

    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'xywh': '%d,%d,%d,%d' % (x, y, w, h)}


def xywh2rect(x, y, w, h):
    x2 = int(x) + int(w)
    y2 = int(y) + int(h)

    return [int(x), int(y), int(x2), int(y2)]


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def frame2npt(frame, fps):
    return frame / fps


def generate_output_path(base, project, video_path):
    temp = re.sub(r'[:/]', '_', video_path)
    if '?' in temp:
        temp = temp.split('?')[0]
    out = os.path.join(base, project, temp)
    os.makedirs(out, exist_ok=True)
    return out

    # load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory, keep_original_size=False, disabled=None):
    if disabled is None:
        disabled = []

    X, y, paths = list(), list(), list()
    proj = directory.rsplit('/')[-1]
    # enumerate folders, on per class
    for subdir in sorted(os.listdir(directory)):
        # path
        path = os.path.join(directory, subdir)
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        files = [os.path.join(path, p) for p in sorted(os.listdir(path))
                 if p != '.DS_Store' and os.path.join(proj, subdir, p) not in disabled]

        faces = [load_gray(file) for file in files]
        if not keep_original_size:
            faces = [resize_img(img) for img in faces]

        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))

        # store
        X.extend(faces)
        y.extend(labels)
        paths.extend(files)

    # Create a list of class names
    class_names = [cls.replace('_', ' ') for cls in np.unique(y)]

    return np.asarray(X), np.asarray(y), paths, class_names


def load_gray(file):
    """Load the image in a 3-channel gray image"""
    # Using PIL instead of cv2.imread because the latter is not working with GIF
    img = np.asarray(Image.open(file).convert('L'))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def fetch_dataset(directory):
    y, paths = list(), list()
    # enumerate folders, on per class
    for subdir in sorted(os.listdir(directory)):
        # path
        path = os.path.join(directory, subdir)
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        files = [os.path.join(path, p) for p in os.listdir(path) if p != '.DS_Store']
        paths.extend(files)

        # create labels
        labels = [subdir.replace('_', ' ') for _ in range(len(files))]
        y.extend(labels)

    return np.asarray(y), paths,


def fix_box(box):
    return [max(0, i) for i in box]  # workaround for https://github.com/ipazc/mtcnn/issues/11


def resize_img(img, image_size=None):
    if image_size is None or img.shape[0] == image_size:
        return img
    scaled = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    scaled = scaled.reshape(-1, image_size, image_size, 3)
    return scaled
