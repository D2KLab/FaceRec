Face-Celebrity-Recognition
==========================

## Install

Install dependencies

    pip install -r requirements.txt
    
If you have errors like `'thread._local' object has no attribute 'value'`, run

    `sh mtcnn_patch.sh`
    
If you want to use also the server capabilities, you need to install [MongoDB](mongodb.com) and run it on default port.

### 1. Building a Training Dataset
Create a directory for raw images utilized for training. In order to download automatically images of celebrity to build the training dataset we need to call the following command:
```sh
python -m src.crawler --keyword "Churchill Winston" --max_num 20 --project proj_name
python -m src.crawler --keyword "Roosevelt Franklin" --max_num 20 --project proj_name
python -m src.crawler --keyword "De Gasperi Alcide" --max_num 20 --project proj_name
```

> NOTE: The crawler is temporarly out of service because of https://github.com/hellock/icrawler/issues/65
 
### 2. Preprocess the raw images (Face detection)
Face alignment using MTCNN
```sh
python -m src.FaceDetector  proj_name --image_size 160
```
### 3. Train a classifier on own images
We perform training a classifier using the following command:
```sh
python -m src.classifier --project proj_name --classifier SVM
```
### 4. Perform face recognition on Video
The below command helps us to recognize people from video using the trained classifier from the previous step:
```sh
python -m src.FaceRecogniser --video video/xxx.mp4 --output_path data/output.txt --project proj_name --video_speedup 1 --folder_containing_frame data/output
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
python -m src.crawler --keyword "Bardot Brigitte" --max_num 20 --project proj_name
```

Then, retrain the model.

> Note: Please empty the data/new_person directory before adding other persons.

### 6. Combine FaceNet + Tracker to perform face recognition on Video

Now we apply **SORT** algorithm to track every face.
**FaceNet** is also applied to every detected frame. 

We track the face from the first frame in which it is detected, and we assign to it the object ID until the tracker lost that ID, and used Facenet to find out the label (the class name) for that ID in all frames having that ID. The Object ID and the temporary label for every face will be generated. After that, the system will try to guess the label for each face by using the majority rule.


Execute the following command:
```sh
python -m src.tracker --video video/xxx.mp4 --project proj_name 
```

         
### 7. Combine Tracker + FaceNet + Cosine Similarity to perform face recognition on Video
We apply **SORT** Tracker to track every face and put them into clusters. Clusters will be generated and stored in `data/out/{video_name}/cluster/{clusterid}`. After that, the system will try to guess the label for each cluster using **majority rule**. A cosine similarity computed between each vector of features in our face training dataset and the ones from each face in our cluster labeled from the previous step is the third step. A cluster is considered to be recognized as a known person if the mean of the three maximum cosine similarities between each face in the cluster and each face in that person training dataset is higher than 0.66 (this value depends on your own dataset, you should do some statistic to pick the fitted one).

Execute the following command (after src.tracker):
```sh
python -m src.clusterize --video video/xxx.mp4 --confidence_threshold 0.7 --dominant_ratio 0.8 --merge_cluster
```

## FaceRec as a service

A service is available as Docker image.

```sh
docker build -t_parser facerec .
docker run -d -p 5050:5000 -p 27017:27017 --name facerec-mongo mongo
docker run -d -p 5050:5000 --restart=unless-stopped  -v /home/semantic/Repositories/Face-Celebrity-Recognition/video:/app/video -v /home/semantic/Repositories/Face-Celebrity-Recognition/data:/app/data -v /home/semantic/Repositories/Face-Celebrity-Recognition/config:/app/config --name facerec1 facerec
```

or 

```
docker-compose up
```

### Special Thanks to:
*  [**Face-Recognition-using-Tensorflow**](https://github.com/davidsandberg/facenet)
*  [**Face-Track-Detect-Extract**](https://github.com/Linzaer/Face-Track-Detect-Extract)

