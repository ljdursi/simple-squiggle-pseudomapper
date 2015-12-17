#!/bin/bash

readonly PLOTSOUT=saveplots
readonly BEST5=ch277_file143
readonly WORST5=ch401_file98
readonly BEST6=ch206_file36
readonly WORST6=ch241_file2

mkdir -p $PLOTSOUT

move() {
    local indir=$1
    local base=$2
    local outdir=$3
    local suffix=$4

    echo "mv $indir/*${base}*template.png ${outdir}/${base}_${suffix}"
    mv $indir/*${base}*template.png ${outdir}/${base}_${suffix}.png
}

./index-and-map.sh
for file in $BEST5 $WORST5 $BEST6 $WORST6; do move plots $file $PLOTSOUT _simple ; done

./index-and-map.sh templateonly noclosest
for file in $BEST5 $WORST5 $BEST6 $WORST6; do move plots $file $PLOTSOUT _noclosest ; done

./index-and-map.sh templateonly noclosest rescale
for file in $BEST5 $WORST5 $BEST6 $WORST6; do move plots $file $PLOTSOUT _rescale ; done

./index-and-map.sh templateonly noclosest rescale extend
for file in $BEST5 $WORST5 $BEST6 $WORST6; do move plots $file $PLOTSOUT _extend ; done

./index-and-map.sh templateonly noclosest rescale extend longest
for file in $BEST5 $WORST5 $BEST6 $WORST6; do move plots $file $PLOTSOUT _longest ; done
