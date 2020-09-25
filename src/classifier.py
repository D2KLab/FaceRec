import argparse
import csv
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import _fit_binary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from tensorflow.keras.models import load_model

from .utils import utils


class FacerecClassifier:
    """
    Inspired by OneVsRestClassifier of sklearn.
    This version avoid normalisation, which brings misleading results
    """

    def __init__(self, type="SVM"):
        self.type = type
        self.estimators_ = []

    def train(self, X, y):
        # Train classifier
        print('Training classifier')

        if self.type == 'SVM':
            model = SVC(kernel='linear', probability=True)
        elif self.type == 'KNN':
            model = KNeighborsClassifier(n_neighbors=1)
        elif self.type == 'Softmax':
            model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        else:
            model = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=100, n_jobs=-1)

        # x = OneVsRestClassifier(model).fit(X,y)
        label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=1)(delayed(_fit_binary)(
            model, X, column, classes=[
                "not %s" % label_binarizer_.classes_[i], label_binarizer_.classes_[i]])
                                              for i, column in enumerate(columns))

        return self

    def predict_proba(self, X):
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        return Y


def main(classifier='SVM', project='general', discard_disabled="true"):
    embedding_file = os.path.join('data/embedding/', project + '.csv')
    label_file = os.path.join('data/embedding/', project + '_label.csv')
    data_dir = os.path.expanduser(os.path.join('data/training_img_aligned/', project))
    classifier_path = os.path.expanduser(os.path.join('data/classifier', project + '.pkl'))
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)

    disabled_file = os.path.join(data_dir, 'disabled.txt')
    disabled = []
    if discard_disabled == "true" and os.path.exists(disabled_file):
        with open(disabled_file) as f:
            disabled = [i.split('training_img_aligned/')[1] for i in f.read().splitlines() if i]
            print(disabled)

    # load train dataset
    trainX, trainy, paths, class_names = utils.load_dataset(data_dir, disabled=disabled)

    facenet = load_model('./model/facenet_keras.h5', compile=False)
    facenet.load_weights('./model/facenet_keras_weights.h5')

    trainX = [utils.get_embedding(facenet, face_pixels) for face_pixels in trainX]
    trainX = np.asarray(trainX)

    np.savetxt(embedding_file, trainX, delimiter=",")
    with open(label_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(trainy, paths))

    if discard_disabled == "auto":
        print("detecting outliers...")
        trainX, trainy, paths, outliers = filter_outliers(trainX, trainy, paths)
        with open(disabled_file, 'w') as f:
            for x in outliers:
                f.write(x)
                f.write('\n')
            f.close()

    model = FacerecClassifier(classifier).train(trainX, trainy)

    # Saving classifier model
    with open(classifier_path, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_path)


def get_outlier_list(project):
    data_dir = os.path.expanduser(os.path.join('data/training_img_aligned/', project))
    trainX, trainy, paths, class_names = utils.load_dataset(data_dir)

    facenet = load_model('./model/facenet_keras.h5', compile=False)
    facenet.load_weights('./model/facenet_keras_weights.h5')

    trainX = [utils.get_embedding(facenet, face_pixels) for face_pixels in trainX]
    trainX = np.asarray(trainX)

    _, _, path, outliers = filter_outliers(trainX, trainy, paths)
    return outliers


def filter_outliers(x, y, paths, threshold=0.1):
    x = np.array(x)
    y = np.array(y)
    paths = np.array(paths)

    classes = np.unique(y)
    to_exclude = []
    outliers = []
    for c in classes:
        index = np.where(y == c)[0]
        _outliers = detect_outliers(x[index], paths[index], threshold)
        for p in _outliers:
            outliers.append(p)
            to_exclude.append(np.where(paths == p)[0][0])
    return np.delete(x, to_exclude, axis=0), np.delete(y, to_exclude), np.delete(paths, to_exclude), outliers


def detect_outliers(embs, files, threshold=0.1):
    # compute distances
    d = cosine_similarity(embs)

    # search for outliers
    outliers = []

    if len(embs) < 2:
        # full remove
        return files

    while d.std() > threshold:
        m = embs.mean(axis=0)
        diff = np.array([cosine_similarity([x], [m]) for x in embs]).flatten()
        to_delete = np.argmin(diff)
        outliers.append(files[to_delete])
        embs = np.delete(embs, to_delete, 0)
        files = np.delete(files, to_delete, 0)
        d = cosine_similarity(embs)

        if len(embs) < 2:
            # full remove
            return files

    return outliers


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str,
                        choices=['KNN', 'SVM', 'RF', 'Softmax'],
                        help='The type of classifier to use.',
                        default='SVM')
    parser.add_argument('--project', type=str, default='general',
                        help='Name of the collection to be part of')
    parser.add_argument('--discard_disabled', default="false",
                        help='If "true", skip the images in the file "disabled.txt". '
                             'If "auto", automatically detect and discard outliers')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.classifier, args.project, args.discard_disabled)
