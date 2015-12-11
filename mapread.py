#!/usr/bin/env python
"""
Using kd-tree index, map ONT fast5 reads to bacterial genomes

 Jonathan Dursi, Jonathan.Dursi@oicr.on.ca
 SimpsonLab, OICR
"""
import argparse
import numpy
import scipy.stats
import matplotlib.pylab
import fast5
import em_rescale
import os
import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle
from spatialindex import KDTreeIndex

def main():
    """
    Driver program for mapping reads.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs="+", type=str, help="Input file[s] to map")
    parser.add_argument('-x', '--suffix', type=str, default="", help="suffix for plot file names")
    parser.add_argument('-D', '--plotdir', type=str, default=".", help="directory for plots")
    parser.add_argument('-r', '--rescale', action="store_true",
                        help="Rescale read based on EM method (quite expensive)")
    parser.add_argument('-e', '--extend', action="store_true",
                        help="Map based on fixed-length extended seeds found")
    parser.add_argument('-b', '--binsize', type=int, default=10000,
                        help="bin size for approximate locations on reference")
    parser.add_argument('-T', '--templateindex', type=str,
                        help="Comma delimited list of index file(s): template")
    parser.add_argument('-C', '--complementindex', type=str, default=None,
                        help="Comma delimited list of index file(s): complement")
    parser.add_argument('-t', '--templateascomplement', action="store_true",
                        help="Use first template index as additional complement index")
    parser.add_argument('-d', '--maxdist', type=float, default=4.25,
                        help="Use dmers within this distance")
    parser.add_argument('-c', '--closest', action="store_true",
                        help="Use only closest dmers")
    parser.add_argument('-p', '--plot', choices=["display", "save", "stats-only"],
                        help="Plot results; default=display", default="display")
    args = parser.parse_args()

    binsize = args.binsize // 2

    # Because the signal-level event corresponding to, eg, TTTTT is not
    # in general easily computable from that of the reverse complement, AAAAA,
    # we need a separate index for the complement strand if we're going to use
    # it; and in fact, there are typically two different possible complement pore
    # models for ONT data.
    args.usecomplement = False
    if args.complementindex is not None or args.templateascomplement:
        args.usecomplement = True

    templ_idxs = [pickle.load(open(filename, "rb"))
                  for filename in args.templateindex.split(',')]
    compl_idxs = []

    if args.complementindex is not None:
        compl_idxs = compl_idxs + [pickle.load(open(filename, "rb"))
                                   for filename in args.complementindex.split(',')]
    if args.templateascomplement:
        compl_idxs.append(templ_idxs[0])

    # reference length
    reflen = templ_idxs[0].reference_length

    # number of bases per dmer = k+dimension-1
    dim = templ_idxs[0].dimension

    for infile in args.infile:
        try:
            reads_templ, maps_templ, readlen = reads_maps_from_fast5(infile, templ_idxs,
                                                                     args.maxdist, args.closest,
                                                                     emrescale=args.rescale)
            reads_compl, maps_compl, _ = reads_maps_from_fast5(infile, compl_idxs,
                                                               args.maxdist, args.closest,
                                                               complement=True,
                                                               emrescale=args.rescale)
            maps_compl = maps_compl + [None]

        except KeyError:
            print("  could not read file "+infile+"; continuing")
            continue

        bin_edges = numpy.arange(-reflen, reflen+binsize-1, binsize)

        # For each set of mappings for the current read, calculate the
        # weighted scores for starting in each bin on the reference
        best = []
        if not args.extend:
            templ_bins = [start_bin_scores(template, binsize) for template in maps_templ]
            compl_bins = [start_bin_scores(complement, binsize) for complement in maps_compl]
        else:
            templ_bins = [start_bin_scores_extension(template, binsize, dim) for template in maps_templ]
            compl_bins = [start_bin_scores_extension(complement, binsize, dim) for complement in maps_compl]

        # Find the combination of template+complement models which
        # produce the highest-scoring localizations
        nbest = 10
        for i, template in enumerate(templ_bins):
            for j, complement in enumerate(compl_bins):
                scores = template+complement

                # overlapping bins
                scores = scores[:-1] + scores[1:]

                # "best" here is zscore: number of std deviations above mean
                zscore = (scores - numpy.mean(scores))/numpy.std(scores)
                localbest = zscore.argsort()[-nbest:][::-1]

                best = best + [(zscore[idx], idx, (bin_edges[idx], bin_edges[idx+2]),
                                i, j, scores) for idx in localbest]

        best.sort()
        best = best[-nbest:][::-1]
        print(os.path.splitext(infile)[0]+" top locations: "+
              str([(b[0], b[2], b[3], b[4]) for b in best]))

        zscore, bestidx, map_range, idx_templ, idx_compl, scores = best[0]
        mappings_templ = maps_templ[idx_templ]
        mappings_compl = maps_compl[idx_compl] if args.usecomplement else None

        allmappings = mappings_templ
        if mappings_compl is not None:
            allmappings = mappings_templ.append(mappings_compl)

        strand = "template"
        if args.usecomplement:
            strand = "template+complement"

        print("%s %s Best location in bin %d,%d zscore = %f,%f" %
              (os.path.splitext(infile)[0], strand, bin_edges[bestidx],
               bin_edges[bestidx+2], zscore, scores[bestidx]))

        if not args.plot == "stats-only":
            matplotlib.rcParams['font.size'] = 8
            matplotlib.pylab.title(infile+": "+strand)
            matplotlib.pylab.plot(bin_edges[bestidx], scores[bestidx], 'ro')
            matplotlib.pylab.plot(bin_edges[1:-1], scores, '.')
            matplotlib.pylab.xlim([-reflen, reflen])

        if args.plot == "save":
            figname = args.plotdir+"/"+os.path.basename(os.path.splitext(infile)[0])+\
                    "-"+strand+args.suffix+".png"
            matplotlib.pylab.savefig(figname)

        elif args.plot == "display":
            matplotlib.pylab.show()

        if not args.plot == "stats-only":
            matplotlib.pylab.clf()


def reads_maps_from_fast5(infile, indexes, maxdist, closest,
                          complement=False, emrescale=False):
    """
    Given a list of indexes and a fast5 filename, read the
    file, scale it to the indices, and perform a lookup on each.

    Input: fast5 filename, list of spatial indices,
            the maximum distance and closest flag for lookup
    Returns: list of scaled reads
             list of mappings
             read length
    """
    read, times, _, _ = fast5.readevents(infile, complement=complement)
    readlen = len(read)

    if emrescale:
        scaledreads = []
        for index in indexes:
            if index is None:
                continue
            model = index.model
            level_means = model.means()
            level_sds = model.sds()
            level_kmers = model.kmers()
            shift, scale, drift, _, _ = em_rescale.rescale(read, times, level_means, level_sds, level_kmers)
            scaledreads.append((read - shift - drift*times)/scale)
    else:
        scaledreads = [index.scale_events(read) for index in indexes
                       if index is not None]
    mappings = [index.lookup(index.events_to_dmer_array(scaledread, each_dmer=True),
                             maxdist=maxdist, closest=closest)
                for scaledread, index in zip(scaledreads, indexes) if index is not None]

    for mapping in mappings:
        if mapping is not None:
            mapping.set_complement(complement)

    return scaledreads, mappings, readlen

def mapping_scores(mappings):
    """
    Returns a score for each mapping 
    """
    if mappings is None:
        return numpy.array([])

    contributions = scipy.stats.norm.pdf(mappings.dmers, mappings.nearest_dmers)
    contributions = numpy.product(contributions, 1)/mappings.nmatches
    totals = collections.defaultdict(float)
    for loc, val in zip(mappings.read_locs, contributions):
        totals[loc] += val
    for i, loc in enumerate(mappings.read_locs):
        contributions[i] /= (totals[loc]+1.e-9)

    return contributions

def start_bin_scores(mappings, binsize, map_range=None, return_bins=False):
    """
    Returns a bin of scores representing a liklihood of the read beginning
    within that bin of reference positions.
    """
    if mappings is None:
        return 0.

    reflen = mappings.reflen
    if map_range is None:
        map_range = [-reflen, reflen+1]

    rangemin = map_range[0]
    rangemax = map_range[1]

    contributions = mapping_scores(mappings)
    starts = mappings.starts

    scorebins = numpy.arange(rangemin, rangemax+binsize-1, binsize)
    scores, _ = numpy.histogram(starts, bins=scorebins, weights=contributions)

    if return_bins:
        return scores, scorebins
    else:
        return scores

def start_bin_scores_extension(mappings, binsize, dim,
                               nextend=3, nskip=1,
                               skip_prob=0.2, stay_prob=0.1,
                               map_range=None, return_bins=False):
    """
    Returns a bin of scores representing a liklihood of the read beginning
    within that bin of reference positions.
    """
    if mappings is None:
        return 0.

    reflen = mappings.reflen
    if map_range is None:
        map_range = [-reflen, reflen+1]

    rangemin = map_range[0]
    rangemax = map_range[1]

    contributions = mapping_scores(mappings)
    move_prob = 1.-skip_prob-stay_prob

    lookup = collections.defaultdict(lambda: collections.defaultdict(float))

    for readpos, refpos, start, score in zip(mappings.read_locs, mappings.idx_locs, mappings.starts, contributions):
        lookup[readpos][refpos] = score
    
    def findextension(readpos, refpos, nextend, nskip):
        val = lookup[readpos][refpos]
        if val == 0. or nextend == 0:
            return val
        delta_ref = +1 if refpos > 0 else -1
        delta_read = +1 if readpos > 0 else -1
        move = findextension(readpos+delta_read*1, refpos+delta_ref*1, nextend-1, nskip)*move_prob
        skip = 0 if nskip == 0 else findextension(readpos+delta_read*(dim-1), refpos+delta_ref*dim, nextend-1, nskip-1)*skip_prob
        stay = 0 if nskip == 0 else findextension(readpos+delta_read*dim, refpos+delta_ref*(dim-1), nextend-1, nskip-1)*stay_prob
        best = max([move, skip, stay])
        return min(best,val)

    extendedscores = [(start, findextension(readpos, refpos, nextend, nskip))
                      for readpos, refpos, start in zip(mappings.read_locs, mappings.idx_locs, mappings.starts)]
    extendedscores = filter(lambda x:x[1] > 0., extendedscores)

    if len(extendedscores) == 0:
        starts = mappings.starts
        print("Warning - could not generate sufficiently large extensions")
    else:
        starts, contributions = zip(*extendedscores)

    scorebins = numpy.arange(rangemin, rangemax+binsize-1, binsize)
    scores, _ = numpy.histogram(starts, bins=scorebins, weights=contributions)

    if return_bins:
        return scores, scorebins
    else:
        return scores


if __name__ == "__main__":
    main()
