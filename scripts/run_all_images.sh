#!/bin/bash

IMAGES=("baby" "cat" "cat2" "dude" "glasses" "lady" "old" "punch" "starbucks")

cd build && make -j8 portrait && cd ../

for img in ${IMAGES[@]} ; do
    echo "Executing image: $img."
    ./build/portrait -l "data/$img-left.jpg" -r "data/$img-right.jpg" -s "data/$img-seg.png" -o "output/$img-portrait.jpg" -d "output/$img-depth.jpg"
done
