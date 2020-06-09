import argparse
import csv
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import load_model

from .utils import utils


def main(classifier='SVM', project='general', discard_disabled=True):
    embedding_file = os.path.join('data/embedding/', project + '.csv')
    label_file = os.path.join('data/embedding/', project + '_label.csv')
    data_dir = os.path.expanduser(os.path.join('data/training_img_aligned/', project))
    classifier_path = os.path.expanduser(os.path.join('data/classifier', project + '.pkl'))
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)

    disabled = []
    if discard_disabled:
        with open(os.path.join(data_dir, 'disabled.txt')) as f:
            disabled = [i.split('training_img_aligned/')[1] for i in f.read().splitlines() if i]

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

    # Train classifier
    print('Training classifier')
    if classifier == 'SVM':
        model = SVC(kernel='linear', probability=True)
    elif classifier == 'KNN':
        model = KNeighborsClassifier(n_neighbors=1)
    elif classifier == 'Softmax':
        model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    else:
        model = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=100, n_jobs=-1)

    model.fit(trainX, trainy)

    # Saving classifier model
    with open(classifier_path, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str,
                        choices=['KNN', 'SVM', 'RF', 'Softmax'],
                        help='The type of classifier to use.',
                        default='SVM')
    parser.add_argument('--project', type=str, default='general',
                        help='Name of the collection to be part of')
    parser.add_argument('--discard_disabled', default=False, action='store_true',
                        help='If true, skip the images in the file "disabled.txt"')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.classifier, args.project, args.discard_disabled)
