from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import argparse
import utils.facenet as facenet
import os
import math
import pickle
import csv
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def main():
    args = parse_args()
    align_command = 'python FaceDetector.py ' + args.input_dir + args.align_dir + ' --image_size 182' + ' --margin 44'
    os.system(align_command)
    print("-------- Alignment Completed ----------")	
    with tf.Graph().as_default():

        with tf.Session() as sess:
            np.random.seed(666)
            datadir = args.align_dir
            embeddingdir = "data/new_person_embedding/"
            modeldir = args.model_path
            
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print(labels)
            
            # # Create a list of class names
            #class_names = [cls.name.replace('_', ' ') for cls in dataset]
            #label_name = [class_names[i] for i in labels]
                            
            print('Number of classes: {}'.format(len(dataset)))
            print('Number of images: {}'.format(len(paths)))
            print('Loading feature extraction model')
            
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            print(embedding_size)

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 200
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                print('{}/{}'.format(i,nrof_batches_per_epoch))
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            # store embedding and labels
            np.savetxt(embeddingdir+'embedding.csv', emb_array, delimiter=",")       
            with open(embeddingdir+'label.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(labels, paths))
            # merge 2 embedding files
            merge_embedding_files("data/embedding/", embeddingdir, "embedding.csv")
            merge_label_files("data/embedding/", embeddingdir, "label.csv")
			
			# re-train the classifier
            start = time.time()
            fname = "data/embedding/label.csv"
            labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
            labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
            labels = list(labels)
            fname = "data/embedding/embedding.csv"
            embeddings = pd.read_csv(fname, header=None).as_matrix()
            le = LabelEncoder().fit(labels)
            class_names = list(le.classes_) 
            class_names = [ i.replace("_"," ") for i in class_names]
            labelsNum = le.transform(labels)
            print(class_names)
            print(labelsNum)
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            print('Start training classifier')
            if(args.classifier == 'SVM'):
                model = SVC(kernel='linear', probability=True)
            elif (args.classifier=='KNN'):
                model = KNeighborsClassifier(n_neighbors=1)
            elif (args.classifier=='Softmax'):
                model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
            elif (args.classifier=='LinearSVC'):
                model = LinearSVC(random_state=0, tol=1e-5)                                                   
            else:
                model = RandomForestClassifier(n_estimators=600, max_depth=420, max_features='auto', n_jobs=-1)   
            model.fit(embeddings, labelsNum)
            print("Re-train the classifier took {} seconds.".format(time.time() - start))
            print('End training classifier')
            print(le)           
            # saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model,class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
	
    parser.add_argument("--input_dir", type=str,
                        help='Path to the data directory containing new person images.',
                        default = "data/new_person" )
    parser.add_argument("--align_dir", type=str,
						help='Path to the data directory containing aligned of new person images.',
						default ="data/aligned_new_person")
    parser.add_argument('--model_path', type=str,
                        help='Path to embedding model',
                        default="model/20180402-114759.pb")
    parser.add_argument('--classifier', type=str, choices=['KNN','SVM','RF','LinearSVC','Softmax'],
                        help='The type of classifier to use.',default='KNN')
    parser.add_argument('classifier_filename',
	                    help='Classifier model file name as a pickle (.pkl) file. ' + 
						'For training this is the output and for classification this is an input.')    
    args = parser.parse_args()
    return args

def merge_embedding_files(old_dir, new_dir, file_name):
	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	next(f) # skip the header
	for line in f:
		fout.write(line)
	f.close() # not really needed

# function to create a map (label, person_name)
def create_map(old_dir, file_name):
	fout = open(old_dir + file_name, "r")
	f_map = open(old_dir + "maps.csv", "w")
	labels = fout.readlines()
	prev_label = ""
	for line in labels:
		curr_label = line.split(',')[0] # get the current label
		if curr_label != prev_label:
			person_name = line.split(',')[1].split('/')[-2] # get the person name
			f_map.write(curr_label + ',' + person_name + '\n')
			prev_label = curr_label
	fout.close()
	f_map.close()

def merge_label_files(old_dir, new_dir, file_name):
	# check if there is already a map or not
	if not os.path.isfile(old_dir + "maps.csv"):
		create_map(old_dir, file_name)

	# get the map from file
	f_map = open(old_dir + "maps.csv", "r")
	maps = f_map.readlines()
	f_map.close()

	fout = open(old_dir + file_name, "a")
	f = open(new_dir + file_name)
	next(f) # skip the header
	prev_label = ""
	save_label = ""
	for line in f:
		split_line = line.split(',')
		if (split_line[0] == prev_label):
			fout.write(str(save_label) + ',' + split_line[1])
			continue
		person_name = split_line[1].split('/')[-2]
		label = [s for s in maps if person_name in s]
		if not label: # this is new person
			label = int(maps[-1].split(',')[0]) + 1
			maps.append(str(label) + ',' + person_name + '\n')
		else:
			label = label[0].split(',')[0]
		fout.write(str(label) + ',' + split_line[1])
		save_label = label
	f.close() # not really needed

	# write back the map to file
	f_map = open(old_dir + "maps.csv", "w")
	f_map.writelines(maps)
	f_map.close()
	
if __name__ == '__main__':
    main()
