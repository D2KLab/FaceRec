# Face-Celebrity-Recognition
### 1. Create a raw image directory
Create a directory for raw images utilized for training such that images of different persons are in different subdirectories. The names of images do not matter, 
and each person can have a different number of images. The images should be formatted as jpg or png and have a lowercase extension.
```sh
$ tree data/raw
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png
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
