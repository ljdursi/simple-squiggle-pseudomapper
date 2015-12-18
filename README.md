# Simple Squiggle Pseudomapper

Simple demonstration and testbed of using spatial indexes (here kd-trees)
to approximately map nanopore squiggle reads.

Running the `index_and_map.sh` script provides a quick demonstration,
indexing the reference sequence of an ecoli strain and approximately 
mapping 10 reads of older and newer nanopore reads to the reference.

Ideas are discussed on the Simpsonlab blog post, http://simpsonlab.github.io/2015/12/18/kdtree-mapping/
