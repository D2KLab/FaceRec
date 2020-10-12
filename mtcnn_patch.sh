#!/bin/sh
b=$(pip show mtcnn | grep Location)
loc="$(cut -d':' -f2 <<<$b)"
loc=$(echo $loc | tr -d ' ')
loc+=/mtcnn/network/factory.py
sed -i '' 's/from keras/from tensorflow.keras/g' $loc
