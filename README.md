# Face-Celebrity-Recognition
### 1. Building a Training Dataset
Create a directory for raw images utilized for training. In order to download automatically images of celebrity to build the training dataset we need to call the following command:
```sh
python crawler.py --keyword "Franco Francesco" --max_num 20 --image_dir data/img_for_training_gg/FrancoFrancesco
```
### 2. Preprocess the raw images (Face detection)
Face alignment using MTCNN
```sh
python FaceDetector.py  data/training_raw_images data/aligned_face/6_persons --image_size 182 --margin 44
```
### 3. Train a classifier on own images
We perform training a classifier using the following command:
```sh
python classifier.py TRAIN --classifier SVM data/aligned/6_persons model/20180402-114759.pb classifier/11_7_2019/svm_classifier_for_6_persons.pkl --batch_size 200
```
### 4. Perform face recognition on Video
The below command helps us to recognize people from video using the trained classifier from the previous step:
```sh
python FaceRecogniser.py --video_dir video/ --output_path BrigitteBardot_2.txt --model_path model/20180402-114759.pb --classifer_path classifier/11_7_2019/svm_classifier_for_6_persons.pkl --video_speedup 1 --folder_containing_frame data/BrigitteBardot_2
```
### 5. Combine FaceNet + Tracker to perform face recognition on Video
Now we apply **SORT** algorithm to track every face. The **FaceNet** is also applied to every detected frame. 
We tracked the face from the beginning it detected and assigned it to the object ID until the tracker lost that ID, and used Facenet to fnd out the label (the class name) for that ID in
all frames having that ID. The Object ID and the temporary label for every face will be generated. After that, the system will try to
guess the label for each face by using the majority rule.
</br>
</br>Execute the following commands in order:
```sh
python Tracker_FaceNet_export_mappingfile.py --video_dir video/ --output_path data/cluster/ --all_trackers_saved all_trackers_saved_BrigitteBardot_1.txt --obid_mapping_classnames obid_mapping_classnames_BrigitteBardot_1.txt --classifer_path classifier/11_7_2019/svm_classifier_for_6_persons.pkl --model_path model/20180402-114759.pb
```
```sh
 python Tracker_FaceNet_export_frames.py --video_dir video --folder_containing_frame data/BrigitteBardot --obid_mapping_classnames_file obid_mapping_classnames_BrigitteBardot_1.txt --output_path data/cluster --classifer_path classifier/11_7_2019/svm_classifier_for_6_persons.pkl --model_path model/20180402-114759.pb --final_output_name_frame_bounding_box BrigitteBardot_1.txt
```
### 6. Combine Tracker + FaceNet + Cosine Similarity to perform face recognition on Video
We apply **SORT** Tracker to track every face and put them into clusters. Clusters will be generated and stored in **"data/cluster/{video_name}/clusterid"**. After that, the system will try to guess the label for each cluster using **majority rule**. A cosine similarity computed between each vector of features in our face training dataset and the ones from each face in our cluster labled from the previous step is the third step. A cluster is considered to be recognized as a known person if the mean of the three maximum cosine similarities between each face in the cluster and each face in that person training dataset is higher than 0.62.
</br>
</br>Execute the following command:
```sh
python Tracker_FaceNet_Making_Clusters.py --video_dir video/ --frame_interval 1 --threshold 0.7 --output_path data/cluster/ --classifer_path classifier/gg/svm_classifier_for_6_persons.pkl --model_path model/20180402-114759.pb --dominant_ratio 0.8 --merge_cluster 1
```
