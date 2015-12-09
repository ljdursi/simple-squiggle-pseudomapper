#!/usr/bin/env python
"""
Contains a SpatialIndex class and an implementation of a KDTreeIndex subclass,
along with a driver program for indexing a reference fasta with the KDTreeIndex.
"""
from __future__ import print_function
import argparse
import numpy
import collections
import poremodel
try:                              #python2: use cPickle instead of pickle
    import cPickle as pickle
except ImportError:
    import pickle

import scipy.spatial
import time
import sys

class Timer(object):
    """
    simple context-manager based timer
    """
    def __init__(self, name, verbose):
        self.__name = name
        self.__verbose = verbose
        self.__start = None

    def __enter__(self):
        self.__start = time.time()
        if self.__verbose:
            print(self.__name + "...", file=sys.stderr)
        return self

    def __exit__(self, *args):
        interval = time.time() - self.__start
        if self.__verbose:
            print(self.__name + ": " + interval + "s", file=sys.stderr)

def readfasta(infilename):
    """
    Reads a fasta file to index, returns a zipped list of labels and sequences
    """
    labels = []
    sequences = []

    curlabel = None
    cursequence = ""

    def updatelists():
        "maintain the sequences and label lists"
        if len(cursequence) is not 0:
            sequences.append(cursequence)
            if curlabel is not None:
                labels.append(curlabel)
            else:
                labels.append('seq'+str(len(sequences)))

    with open(infilename, 'r') as infile:
        for line in infile:
            if line[0] == ">":
                updatelists()
                cursequence = ""
                curlabel = line[1:].strip()
            else:
                cursequence += line.strip()

    updatelists()
    return list(zip(labels, sequences))

def reverse_complement(seq):
    """ reverse complement of a sequence """
    rcbases = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N'}
    return "".join([rcbases[base] if base in rcbases else base for base in seq[::-1]])

def kmer_to_int64(kmer):
    """
    Convert a kmer into a numerical rank

    input:      kmer (string) - sequence
    returns:    numpy.int64 rank
    """
    vals = {'A':0, 'C':1, 'G':2, 'T':3}
    nalphabet = 4

    val = 0
    for base in kmer:
        if not base in vals:
            return numpy.int64(-1)
        val = val * nalphabet + vals[base]
    return numpy.int64(val)

class SpatialIndex(object):
    """
    The SpatialIndex class defines the interface for a spatial index
    on a genome given a pore model (which maps kmers to floating point values).

    The spatial index requires a dimension in which to map, a pore model,
    and the sequence to index.

    The SpatialIndex class lacks a complete implementation; the KDTreeIndex
    is a subclass which completely implements the interface.  The SpatialIndex
    class factors out that functionality which isn't specific to the k-d tree
    index (eg, would be needed for an r-tree implementation).
    """
    def __init__(self, sequence, name, model, dimension, maxentries=10):
        """
        Initialization:

        sequence:  sequence to index
        name:      name of the index
        model:     a pore model
        dimension: dimension in which to do the spatial index
        maxentries:filter out any uniformitive points
                    (those that occur more than this many times in the sequence)
        """
        self.__max_locations = maxentries
        self.__model = model
        self.__dimension = dimension
        self.__referencename = name
        self.__reflen = len(sequence)
        self.__locations = None
        self.__starts = None

    def set_locs_and_starts(self, locations, starts):
        """
        Sets the matching locations of each individual kmer and where
        each unique kmer position starts in that list; do it in a method
        here so it is set in the superclass
        """
        self.__locations = locations
        self.__starts = starts

    @classmethod
    def kmers_from_sequence(cls, sequence, model, dimension,
                            include_complement=True, maxentries=10, verbose=False):
        """
        This routine, called on initialization, reads in the sequence to index
        and generates three key arrays used in the lookup routines:
            - the list of unique dmers (expressed as int64s, their rank: eg, AAA...A = 0).
            - A list of genomic locations in the reference
            - A list of starts, so that the ith lexicographic dmer present in the
              sequence corresponds to locations [starts[i],starts[i+1])
        """
        if not isinstance(model, poremodel.PoreModel):
            raise ValueError("KDTreeIndex.__init__(): must provide pore model")

        # First, the sequence (and its reverse complement) is turned into:
        #  - an array of integers representing dmers in sequence (eg, AA..A = 0)
        #  - the list of genomic locations corresponding to each
        #     ( which is just 1,2,3...N,-1,-2,-3...-N )
        #    with negative values corresponding to locations on the complement
        #    strand

        # convert to events (which represent kmers)
        with Timer("Generating events", verbose) as timer:
            complement = ""
            if include_complement:
                complement = reverse_complement(sequence)

            refevents, refsds = model.sequence_to_events(sequence)
            compevents, compsds = model.sequence_to_events(complement)

            allkmers = numpy.zeros(len(refevents)-dimension+1+
                                   len(compevents)-dimension+1,
                                   dtype=numpy.int64)

        # convert k+d-1-mers into integers
        with Timer("Converting to integer representations", verbose) as timer:
            # this could fairly easily be sped up
            for i in range(len(compevents)-dimension+1):
                allkmers[i] = kmer_to_int64(complement[i:i+model.k+dimension-1])

            shift = len(compevents)-dimension
            for i in range(len(refevents)-dimension+1):
                allkmers[i+shift] = kmer_to_int64(sequence[i:i+model.k+dimension-1])

        # sort by value of the dmer integer values
        with Timer("Sorting", verbose) as timer:
            locs = numpy.concatenate((-numpy.arange(len(compevents)-dimension+1)-1,
                                      numpy.arange(len(refevents)-dimension+1)+1))
            positions = allkmers.argsort()
            allkmers, locs = allkmers[positions], locs[positions]
            del positions

            # get rid of invalid dmers (eg, containing N) which return -1
            start = numpy.argmax(allkmers >= 0)
            allkmers, locs = allkmers[start:], locs[start:]

        # generate list of unique dmers (as integers) and start locations
        with Timer("Building Counts", verbose) as timer:
            kmers, counts = numpy.unique(allkmers, return_counts=True)
            starts = numpy.cumsum(counts)
            starts = numpy.concatenate((numpy.array([0]), starts))
            if verbose:
                print("total entries = "+str(len(kmers)))

        # extract the dmer level means and std deviations corresponding to each dmer
        with Timer("Extracting Kmers", verbose) as timer:
            data = numpy.zeros((len(kmers), dimension), dtype=numpy.float32)
            sdata = numpy.zeros((len(kmers), dimension), dtype=numpy.float32)

            for i in range(len(kmers)):
                loc = locs[starts[i]]
                if loc < 0:
                    idx = -(loc+1)
                    events = compevents[idx:idx+dimension]
                    sds = compsds[idx:idx+dimension]
                else:
                    idx = loc-1
                    events = refevents[idx:idx+dimension]
                    sds = refsds[idx:idx+dimension]

                data[i, :] = numpy.array(events, numpy.float32)
                sdata[i, :] = numpy.array(sds, numpy.float32)

        return data, sdata, locs, starts

    @property
    def dimension(self):
        """Returns the spatial dimension of the index"""
        return self.__dimension

    @property
    def reference_length(self):
        """Returns the size of the sequence indexed"""
        return self.__reflen

    def index_to_genomic_locations(self, idxs):
        """
        Returns a list-of-list of genomic locations corresponding to each
        dmer index (index into the sorted list of unique dmers in the reference)
        in idxs
        """
        def idx_to_locs(idx):
            """returns list of locations for one index"""
            return self.__locations[self.__starts[idx]:self.__starts[idx+1]]

        if isinstance(idxs, int) or isinstance(idxs, numpy.int64):
            return list(idx_to_locs(idxs))
        else:
            return [idx_to_locs(idx) for idx in idxs]

    def events_to_dmer_array(self, events, each_dmer=False, each_event=False):
        """
        Convert a 1d array of events to an array of d-dimensional points for
        the index

        inputs: events - 1d array of events
                each_dmer, each_event: booleans, one of which must be true
                each_dmer: every unique dmer is returned
                each_event: dmers are returned s/t every event occurs once

        outputs: 2d array of dmers
                 if each_dmer is true array is of size (|events|-dimension+1, d)
                 if each_event is true array is of size (|events|//dimension, d)
        """
        if (each_dmer and each_event) or not(each_dmer or each_event):
            raise ValueError("events_to_dmer_array: one of each_kmer, each_event must be True")

        dim = self.__dimension
        if each_event:
            step = dim
        else:
            step = 1

        kmers = numpy.array([events[i:i+dim]
                             for i in range(0, len(events)-dim+1, step)])
        return kmers

    def scale_events(self, events, sds=None):
        """
        Scales the events to be consistent with the pore model
        """
        return self.__model.scale_events(events, sds)

    def lookup(self, read_dmers, maxdist=4.25, closest=False):
        """ Lookup function """
        pass

    @property
    def model(self):
        """Return the pore model"""
        return self.__model

class Mappings(object):
    """
    The spatial indices return the lookups as mappings, from read
    locations to genomic locations.  This is more complicated than
    in the base-called case, as any signal-level dmer may well
    correspond to multiple basecalled dmers, which may each occur
    in various locations in the sequence.

    Objects of the mappings class contain:
        information about the read and the reference, and whether this
          is a complement-strand index
        an array of read locations, with repeats
        an array of corresponding reference locations
        for each read loc - ref loc, the distance to the proposed dmer

    The class defines several operations on these mappings.
    """
    def __init__(self, readlocs, idxlocs, dists, dmers, nearestdmers, nmatches,
                 referenceLen, readlen, complement=False):
        """
        Initializes a mapping.
        Inputs:
            - array of read locations
            - same-sized array of reference locations
                (+ve for template strand, -ve for complement strand)
            - same-sized array of distances (from read dmer to proposed ref dmer)
            - same-sized array of the proposed signal-level dmer
        """
        assert len(readlocs) == len(idxlocs)
        assert len(idxlocs) == len(dists)
        assert nearestdmers.shape[0] == len(dists)
        self.read_locs = readlocs
        self.idx_locs = idxlocs
        self.dists = dists
        self.dmers = dmers
        self.nearest_dmers = nearestdmers
        self.nmatches = nmatches
        self.reflen = referenceLen
        self.readlen = readlen
        self.complement = complement

    def __str__(self):
        """
        Readable output summary of a mapping
        """
        output = "Mappings: ReferenceLength = %d, complement = %d, readlength = %d\n" \
                % (self.reflen, self.readlen, self.complement)
        output += "        : nmatches = %d\n" % (len(self.read_locs))
        for read, idx, dist, nearest in zip(self.read_locs[:5], self.idx_locs[:5],
                                            self.dists[:5], self.nearest_dmers[:5, :]):
            output += " %d: %d (%5.3f) " % (read, idx, dist) + \
                      numpy.array_str(nearest, precision=2, suppress_small=True) + "\n"
        if len(self.read_locs) > 5:
            output += " ...\n"
        return output

    def set_complement(self, complement=True):
        """
        Sets the complement strand flag of the mapping
        """
        self.complement = complement

    def coords_compl_to_templ(self, complement_coords):
        """
        Converts a set of complement strand coordinates to corresponding
        template strand coordinates, given the mapping reference length
        """
        # if a complement strand maps to |pos| on the complement of reference -> -pos,
        # then template strand would map to reflen-|pos| on template strand of ref = ref+pos;
        # if a complement strand maps to pos on the template of the reference -> +pos,
        # then the template strand would map to (ref-|pos|) of the complement = -ref+pos
        return -numpy.sign(complement_coords)*self.reflen+complement_coords

    @property
    def starts(self):
        """
        Returns the array of implied starting positions of the read, given the
        list of read-to-reference mappings
        """
        startlocs = numpy.sign(self.idx_locs)*\
                     numpy.mod(numpy.abs(self.idx_locs)-self.read_locs+self.reflen, self.reflen)
        if self.complement:
            startlocs = self.coords_compl_to_templ(startlocs)
        return startlocs

    def append(self, other):
        """
        Returns a new set of mappings consisting of the current
        mappings data and another set appended to it.
        If the two are of the same strand, this is a simple concatenation;
        if not, must convert to the template coordinates and append
        """
        if self.complement == other.complement:
            return Mappings(numpy.concatenate((self.read_locs, other.read_locs)),
                            numpy.concatenate((self.idx_locs, other.idx_locs)),
                            numpy.concatenate((self.dists, other.dists)),
                            numpy.concatenate((self.dmers, other.dmers)),
                            numpy.concatenate((self.nearest_dmers, other.nearest_dmers)),
                            numpy.concatenate((self.nmatches, other.nmatches)),
                            self.reflen,
                            self.readlen+other.readlen,
                            self.complement)
        else:
            template, complement = (self, other) if self.complement else (other, self)
            return Mappings(numpy.concatenate((template.read_locs,
                                               template.readlen-complement.read_locs)),
                            numpy.concatenate((template.idx_locs,
                                               complement.coords_compl_to_templ(complement.idx_locs))),
                            numpy.concatenate((template.dists, complement.dists)),
                            numpy.concatenate((self.dmers, other.dmers)),
                            numpy.concatenate((template.nearest_dmers, complement.nearest_dmers)),
                            numpy.concatenate((self.nmatches, other.nmatches)),
                            template.reflen,
                            template.readlen,
                            False)

    def local_rescale(self, read_dmers, map_range):
        """
        As with scaling read events to a model, in this case we update the
        distances and signal-level dmers by re-scaling to the mappings,
        to improve the accuracy of the original crude rescaling which used
        no information about correspondance between particular read and
        model events.
        """
        reflen = self.reflen
        if map_range is None:
            map_range = (-reflen, reflen+1)

        starts = self.starts
        valid = numpy.where((starts >= map_range[0]) & (starts <= map_range[1]))[0]

        read_events = read_dmers[self.read_locs[valid], :].reshape(read_dmers[self.read_locs[valid], :].size)
        idx_events = self.nearest_dmers[valid, :].reshape(self.nearest_dmers[valid, :].size)

        if valid.size == 0:
            return self, (1, 0)
        fit = numpy.polyfit(read_events, idx_events, 1)
        new_dmers = fit[0]*read_dmers + fit[1]
        dists = numpy.sqrt(numpy.sum((new_dmers[self.read_locs, :] - self.nearest_dmers)*\
                                     (new_dmers[self.read_locs, :] - self.nearest_dmers), axis=1))

        return Mappings(self.read_locs, self.idx_locs, dists, self.dmers, self.nearest_dmers,
                        self.nmatches, reflen, self.readlen, self.complement), fit


class KDTreeIndex(SpatialIndex):
    """
    Specialization of the Spatial Index class which uses kdtrees;
    uses cKDTree in scipy.spatial, with very small modifications
    also works with sklearn.neighbours kdtree
    """
    def __init__(self, sequence, name, model, dimension, include_complement=True,
                 maxentries=10, verbose=False):
        super(KDTreeIndex, self).__init__(sequence, name, model, dimension, maxentries)

        events, _, locations, starts = self.kmers_from_sequence(sequence, model,
                                                                dimension, include_complement,
                                                                maxentries, verbose)
        self.set_locs_and_starts(locations, starts)

        if verbose:
            print("KDTree: building tree")
        #pylint: disable=not-callable
        self.__kdtree = scipy.spatial.cKDTree(events)
        #pylint: enable=not-callable
        #self.__kdtree = sklearn.neighbors.KDTree(events)

    @property
    def reference_length(self):
        """Return reference length"""
        return super(KDTreeIndex, self).reference_length

    @property
    def dimension(self):
        """Return index dimension"""
        return super(KDTreeIndex, self).dimension

    def scale_events(self, events, sds=None):
        """Scale input (read) events to model"""
        return super(KDTreeIndex, self).scale_events(events, sds)

    def lookup(self, read_dmers, maxdist=4.25, closest=False):
        """
        For a given set of read events, return a list of mappings
        between the read events and the indexed reference.

        Inputs:
            - read_kmers: signal-level events from a read
            - maxdist: maximum allowable distance between points in read & ref
            - closest (optional, boolean: default False): if True,
                only return mappings to closest dmer.  If False,
                return mappings to all dmers within maxdist
        """
        Match = collections.namedtuple("Match", ["dist", "idx_locs", "nearestDmer", "readLoc"])

        if read_dmers.ndim == 1:
            read_dmers = [read_dmers]

        def dist(read_posn, idx):
            """Distance between read_dmer at position read_pos and dmer index idx"""
            p = read_dmers[read_posn, :]
            q = self.__kdtree.data[idx, :]
            return numpy.sqrt(numpy.max((p-q)*(p-q)))

        if closest:
            dists, idxs = self.__kdtree.query(read_dmers)
            nearests = self.__kdtree.data[idxs, :]
            matches = [self.index_to_genomic_locations(idx) for idx in idxs]
            results = [Match(*result) for result in zip(dists, matches,
                                                        nearests, range(len(dists)))]
        else:
            idxs = self.__kdtree.query_ball_point(read_dmers, maxdist)
            #below is the corresponding line for sklearn.neighbours.KDTree
            #idxs = self.__kdtree.query_radius(read_dmers, maxdist)
            results = [Match(dist(posn, pidx), self.index_to_genomic_locations(pidx),
                             self.__kdtree.data[pidx, :], posn)
                       for posn, pidxs in enumerate(idxs)
                       for pidx in pidxs]

        dists = numpy.array([match.dist for match in results for idx in match.idx_locs],
                            dtype=numpy.float32)
        read_locs = numpy.array([match.readLoc for match in results for idx in match.idx_locs],
                                dtype=numpy.int)
        idx_locs = numpy.array([idx for match in results for idx in match.idx_locs],
                               dtype=numpy.int)
        nmatches = numpy.array([len(match.idx_locs) for match in results for idx in match.idx_locs],
                               dtype=numpy.int)
        nearests = numpy.array([match.nearestDmer for match in results for idx in match.idx_locs],
                               dtype=numpy.float32)
        dmers = numpy.array([read_dmers[match.readLoc] for match in results for idx in match.idx_locs],
                               dtype=numpy.float32)

        return Mappings(read_locs, idx_locs, dists, dmers, nearests, nmatches,
                        referenceLen=self.reference_length, readlen=len(read_dmers))

    def __getstate__(self):
        """
        Need to modify default get/set state routines
        so that nested cKDTree can be pickled
        """
        kdstate = self.__kdtree.__getstate__()
        self.__kdtree = None
        state = self.__dict__.copy()
        state["_kdtree"] = kdstate
        return state

    def __setstate__(self, state):
        """
        Need to modify default get/set state routines
        so that nested cKDTree can be pickled
        """
        kdstate = state["_kdtree"]
        del state["_kdtree"]
        self.__dict__.update(state)
        #pylint: disable=not-callable
        self.__kdtree = scipy.spatial.cKDTree([[0]*self.dimension, [1]*self.dimension])
        #pylint: enable=not-callable
        self.__kdtree.__setstate__(kdstate)


def main():
    """
    Driver program - generate a spatial index from a reference
    """
    parser = argparse.ArgumentParser(description="Build a KDTree index of reference given a pore model")

    parser.add_argument('reference', type=str, help="Reference FASTA")
    parser.add_argument('modelfile', type=str, help="Pore model file")
    parser.add_argument('outfile', type=str, help="Output file prefix to save (pickled) index")
    parser.add_argument('-D', '--dimension', type=int, default=10,
                        help="Number of dimensions to use in spatial index")
    parser.add_argument('-m', '--maxentries', type=int, default=5,
                        help="Filter out reference dmers that occur in more m locations")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('-C', '--complement', action="store_true",
                        help="Use Complement model rather than Template (from 2D Fast5 files only)")
    parser.add_argument('-d', '--delimiter', default=",", type=str,
                        help="Pore model delimeter (for CSV/TSV files only)")
    parser.add_argument('-w', '--boxwidth', default=2., type=float,
                        help="box size (in std dev) for rtree entries")

    args = parser.parse_args()

    if args.verbose:
        print("Reading Model...")
    pore_model = poremodel.PoreModel(args.modelfile, args.complement, args.delimiter)

    if args.verbose:
        print("Indexing...")

    contigs = readfasta(args.reference)

    for label, ref_sequence in contigs:
        labelbase = label.split()[0]

        if len(contigs) == 1:
            referencename = args.reference
            outfilename = args.outfile+".kdtidx"
        else:
            referencename = args.reference+"-"+labelbase
            outfilename = args.outfile+"-"+labelbase+".kdtidx"

        index = KDTreeIndex(ref_sequence, referencename, pore_model,
                            args.dimension, maxentries=args.maxentries)

        if args.verbose:
            print("Saving Index...")
        with open(outfilename, "wb") as pickle_file:
            pickle.dump(index, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
