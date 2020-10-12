#!/bin/sh
b=$(pip show icrawler | grep Location)
loc="$(cut -d':' -f2 <<<$b)"
loc=$(echo $loc | tr -d ' ')
loc+=/icrawler/builtin/google.py
sed -i '' 's|txt = re.sub(r"^AF_initDataCallback\\({.*key: '"'"'ds:(\\d)'"'"'.+data:function\\(\\){return (.+)}}\\);?$"|txt = re.sub(r"^AF_initDataCallback\\({.*key: '"'"'ds:(\\d)'"'"'.+data:(.+)(, sideChannel: {.*})}\\);?$"|g' $loc
