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


def main(classifier='SVM', data_dir='data/training_img_aligned', classifier_path='data/classifier/classifier.pkl'):
    embedding_dir = "data/embedding/"

    # load train dataset
    trainX, trainy, paths, class_names = utils.load_dataset(data_dir)

    facenet = load_model('./model/facenet_keras.h5', compile=False)
    facenet.load_weights('./model/facenet_keras_weights.h5')

    trainX = [utils.get_embedding(facenet, face_pixels) for face_pixels in trainX]
    trainX = np.asarray(trainX)

    np.savetxt(embedding_dir + 'embedding.csv', trainX, delimiter=",")
    with open(embedding_dir + 'label.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(trainy, paths))

    classifier_filename_exp = os.path.expanduser(classifier_path)
    os.makedirs(os.path.dirname(classifier_filename_exp), exist_ok=True)

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
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str,
                        choices=['KNN', 'SVM', 'RF', 'Softmax'],
                        help='The type of classifier to use.',
                        default='SVM')
    parser.add_argument('--data_dir', type=str, default='data/training_img_aligned',
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--classifier_path', type=str, default='data/classifier/classifier.pkl',
                        help='Path to the KNN classifier')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.classifier, args.data_dir, args.classifier_path)
