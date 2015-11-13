#!/bin/bash 
# demo script for running the simple squiggle quasi-mapper 

REFERENCE=ecoli.fa

function makeindex {
# build a kd-tree index given a reference and a pore model, and a given dimensionality d
    model=$1
    indexfile=$2
    dimension=${3-8}

    if [ ! -f ${indexfile}.kdtidx ]
    then
        echo "Indexing : model $model, dimension $dimension"
        echo "python spatialindex.py --dimension $dimension $REFERENCE $model $indexfile"
        time python spatialindex.py --dimension $dimension $REFERENCE $model $indexfile
    fi
}

function mapreads {
# map a list of reads given the provided indices
    declare -a input=("${!1}")
    output=$2
    template_idx=$3
    complement_idx=$4
    template_as_complement=$5

    if [ ! -f $output ] 
    then
        echo "Mapping reads: starting with ${input}"
        if [ ! -z "$4" ]
        then
            complementoption="--complementindex"
        fi
        if [ ! -z "$5" ]
        then
            template_as_complement="--templateascomplement"
        fi

        echo "time python ./mapread.py --plot save --plotdir plots --closest --maxdist 3 "\
                "--templateindex ${template_idx} "\
                "${complementoption} ${complement_idx}" \
                "$input ...  > $output"

        time python ./mapread.py --plot save --plotdir plots --closest --maxdist 3 \
                --templateindex ${template_idx} \
                ${complementoption} ${complement_idx} \
                ${input[@]} > $output
    fi
}

function niceoutput {
# merge the output of mapreads with the known results to print a nice summary table
    filename=$1
    bwaout=$2
    export LANG=en_EN   # for sort, join

    join <( sort -k1 $bwaout) \
         <(cat ${filename} | grep zscore | sed -e "s#^.*_ch#ch#" -e 's#_strand##' -e 's/,/ /g'| \
            sed -e "s/_strand//" | \
            awk '{size=4641648; mean=int(($7+$8)/2); if (mean < 0) mean=mean+size; printf "%s\t%s\t%d\t%8.3f\n",$1,$2,mean,$11}' | \
            sort -k1) \
       | awk 'function dist(p1,p2) {size=4641648; d=p1-p2; if (d<0) d=-d; if (d*2 > size) d=size-d; return d;} \
              {d=dist($2,$4); printf "%s\tDifference\t%d\tBWA\t%d\t%s\t%d\tzscore\t%f\n",$1,d,$2,"kDTree",$4,$5}' \
       | sort -k3n | column -t
}


mkdir -p plots
mkdir -p indices

echo 
echo "5mer data: "
echo 

makeindex models/5mer/template.model indices/ecoli-5mer-template 8
makeindex models/5mer/complement.model indices/ecoli-5mer-complement 8

FILES=($( ls ecoli/005/*fast5 ))
mapreads FILES[@] template-only-005.txt indices/ecoli-5mer-template.kdtidx 
mapreads FILES[@] template-complement-005.txt indices/ecoli-5mer-template.kdtidx indices/ecoli-5mer-complement.kdtidx usetemplate

echo "Template Only Alignments"
niceoutput template-only-005.txt ecoli/005/bwamem-ecoli-map-locations.txt

echo ""
echo "Template+Complement Alignements"
niceoutput template-complement-005.txt ecoli/005/bwamem-ecoli-map-locations.txt

echo
echo
echo "6mer data: "
echo 

makeindex models/6mer/template.model indices/ecoli-6mer-template 7
makeindex models/6mer/complement_pop1.model indices/ecoli-6mer-complement_pop1 7
makeindex models/6mer/complement_pop2.model indices/ecoli-6mer-complement_pop2 7

FILES=($( ls ecoli/006/*fast5 ))
mapreads FILES[@] template-only-006.txt indices/ecoli-6mer-template.kdtidx
mapreads FILES[@] template-complement-006.txt indices/ecoli-6mer-template.kdtidx\
    indices/ecoli-6mer-complement_pop1.kdtidx,indices/ecoli-6mer-complement_pop2.kdtidx

echo "Template Only Alignments"
niceoutput template-only-006.txt ecoli/006/bwamem-ecoli-map-locations.txt

echo ""
echo "Template+Complement Alignements"
niceoutput template-complement-006.txt ecoli/006/bwamem-ecoli-map-locations.txt
