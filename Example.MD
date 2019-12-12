python crawler.py --keyword "O'Brien Conan" --max_num 20 --image_dir data/training_img/OBrienConan
python crawler.py --keyword "Burr Bill" --max_num 20 --image_dir data/training_img/BurrBill


python FaceDetector.py  data/training_img/ data/training_img_aligned --image_size 182 --margin 44

python classifier.py TRAIN --classifier SVM data/training_img_aligned model/20180402-114759.pb classifier/classifier.pkl --batch_size 200

python FaceRecogniser.py --video_dir video/ --output_path data/output.txt --model_path model/20180402-114759.pb --classifer_path classifier/classifier.pkl --video_speedup 1 --folder_containing_frame data/output

python Tracker_FaceNet_export_mappingfile.py --video_dir video/ --output_path data/cluster/ --all_trackers_saved data/all_trackers_saved.txt --obid_mapping_classnames  data/obid_mapping_classnames.txt --classifer_path classifier/classifier.pkl --model_path model/20180402-114759.pb
