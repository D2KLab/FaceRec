#!/bin/sh
b=$(pip show icrawler | grep Location)
loc="$(cut -d':' -f2 <<<$b)"
loc=$(echo $loc | tr -d ' ')
loc+=/icrawler/builtin/google.py
sed -i '' 's|Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3)|Mozilla/5.0 (Windows NT 10.0; Win64; x64)|g' $loc
sed -i '' 's|Chrome/48.0.2564.116 Safari/537.36)|Chrome/88.0.4324.104 Safari/537.36)|g' $loc
