Face-Celebrity-Recognition
==========================

### 1. Building a Training Dataset
Create a directory for raw images utilized for training. In order to download automatically images of celebrity to build the training dataset we need to call the following command:
```sh
python -m src.crawler --keyword "Churchill Winston" --max_num 20 --image_dir data/training_img/ChurchillWinston
python -m src.crawler --keyword "Roosevelt Franklin" --max_num 20 --image_dir data/training_img/RooseveltFranklin
python -m src.crawler --keyword "De Gasperi Alcide" --max_num 20 --image_dir data/training_img/DeGasperiAlcide
```
### 2. Preprocess the raw images (Face detection)
Face alignment using MTCNN
```sh
python -m src.FaceDetector  data/training_img/ data/training_img_aligned --image_size 182 --margin 44
```
### 3. Train a classifier on own images
We perform training a classifier using the following command:
```sh
python -m src.classifier TRAIN --classifier SVM data/training_img_aligned  --model-path model/20180402-114759.pb --classifier_path classifier/classifier.pkl --batch_size 200
```
### 4. Perform face recognition on Video
The below command helps us to recognize people from video using the trained classifier from the previous step:
```sh
python -m src.FaceRecogniser --video video/xxx.mp4 --output_path data/output.txt --model_path model/20180402-114759.pb --classifier_path classifier/classifier.pkl --video_speedup 1 --folder_containing_frame data/output
```
### 5. Adding new persons (or images of existing persons) into a system
First, creating a directory for raw images of new persons as follow. 
```sh
$ tree data/new_person
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

Please note that for every new person added, you should add as many images of that person as of previous ones.

```sh
python -m src.crawler --keyword "Bardot Brigitte" --max_num 20 --image_dir data/new_person/BardotBrigitte
```

Then, the following command should be executed:
```sh
python -m src.add_new_person  --input_dir data/new_person --align_dir data/new_person_aligned/ --model_path model/20180402-114759.pb --classifier SVM classifier/classifier_2.pkl
```
> Note: Please empty the data/new_person directory before adding other persons.

### 6. Combine FaceNet + Tracker to perform face recognition on Video

Now we apply **SORT** algorithm to track every face.
**FaceNet** is also applied to every detected frame. 

We track the face from the first frame in which it is detected, and we assign to it the object ID until the tracker lost that ID, and used Facenet to find out the label (the class name) for that ID in all frames having that ID. The Object ID and the temporary label for every face will be generated. After that, the system will try to guess the label for each face by using the majority rule.


Execute the following command:
```sh
python -m src.Tracker_FaceNet_export_mappingfile --video video/xxx.mp4 --face_fragment_path data/cluster/ --trackers_csv data/trackers.csv --predictions_csv data/predictions.csv --classifier_path classifier/classifier.pkl --model_path model/20180402-114759.pb --export_frames --frames_path data/frames --trackout_csv data/trackout.csv
```

         
### 7. Combine Tracker + FaceNet + Cosine Similarity to perform face recognition on Video
We apply **SORT** Tracker to track every face and put them into clusters. Clusters will be generated and stored in `data/cluster/{video_name}/clusterid`. After that, the system will try to guess the label for each cluster using **majority rule**. A cosine similarity computed between each vector of features in our face training dataset and the ones from each face in our cluster labeled from the previous step is the third step. A cluster is considered to be recognized as a known person if the mean of the three maximum cosine similarities between each face in the cluster and each face in that person training dataset is higher than 0.66 (this value depends on your own dataset, you should do some statistic to pick the fitted one).

Execute the following command:
```sh
python -m src.Tracker_FaceNet_Making_Clusters --video_dir video/ --frame_interval 1 --threshold 0.7 --output_path data/cluster/ --classifier_path classifier/classifier.pkl --model_path model/20180402-114759.pb --dominant_ratio 0.8 --merge_cluster 1
```
### Special Thanks to:
*  [**Face-Recognition-using-Tensorflow**](https://github.com/davidsandberg/facenet)
*  [**Face-Track-Detect-Extract**](https://github.com/Linzaer/Face-Track-Detect-Extract)
