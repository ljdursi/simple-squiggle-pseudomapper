#!/usr/bin/env python
"""
Using kd-tree index, map ONT fast5 reads to bacterial genomes

 Jonathan Dursi, Jonathan.Dursi@oicr.on.ca
 SimpsonLab, OICR
"""
import argparse
import cProfile
import numpy
import scipy.stats
import matplotlib.pylab
import fast5
import em_rescale
import os
import collections
from bisect import bisect_left
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
    parser.add_argument('-e', '--extend', type=int, default=0,
                        help="Extend found seeds this many times")
    parser.add_argument('-s', '--skips', type=int, default=2,
                        help="Allow this many skips/stays if extending")
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
    parser.add_argument('-l', '--longest', action="store_true",
                        help="Use longest collinear runs rather than binning")
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
            _, maps_templ, readlen = reads_maps_from_fast5(infile, templ_idxs,
                                                           args.maxdist, args.closest,
                                                           emrescale=args.rescale)
            _, maps_compl, _ = reads_maps_from_fast5(infile, compl_idxs,
                                                     args.maxdist, args.closest,
                                                     complement=True,
                                                     emrescale=args.rescale)
            maps_compl = maps_compl + [None]

        except KeyError:
            print("  could not read file "+infile+"; continuing")
            continue


        # For each set of mappings for the current read, calculate the
        # weighted scores for starting in each bin on the reference
        if args.longest:
            templ_bins, templ_posns = zip(*[colinear_extended_matches(template, binsize, dim, 
                                                                      nextend=args.extend, nskip=args.skips)
                                          for template in maps_templ])
            compl_bins, compl_posns = zip(*[colinear_extended_matches(complement, binsize, dim, 
                                                                      nextend=args.extend, nskip=args.skips)
                                          for complement in maps_compl])
        else:
            bin_edges = numpy.arange(-reflen, reflen+binsize-1, binsize)
            templ_bins = [start_bin_scores_extension(template, binsize, dim, 
                                                     nextend=args.extend, nskip=args.skips)
                          for template in maps_templ]
            compl_bins = [start_bin_scores_extension(complement, binsize, dim, 
                                                     nextend=args.extend, nskip=args.skips)
                          for complement in maps_compl]

        # Find the combination of template+complement models which
        # produce the highest-scoring localizations
        nbest = 10
        best = []
        for i, template in enumerate(templ_bins):
            for j, complement in enumerate(compl_bins):
                scores = numpy.array(template+complement)
                if args.longest:
                    posns = numpy.array(templ_posns[i] + compl_posns[j])
                else:
                    posns = bin_edges

                # overlapping bins
                if not args.longest:
                    scores = scores[:-1] + scores[1:]

                # "best" here is zscore: number of std deviations above mean
                zscore = (scores - numpy.mean(scores))/numpy.std(scores)
                localbest = zscore.argsort()[-nbest:][::-1]

                best = best + [(zscore[idx], idx, posns[idx], i, j, scores) for idx in localbest]

        best.sort()
        best = best[-nbest:][::-1]
        print(os.path.splitext(infile)[0]+" top locations: "+
              str([(b[0], b[2], b[3], b[4]) for b in best]))

        zscore, bestidx, _, _, _, scores = best[0]

        strand = "template"
        if args.usecomplement:
            strand = "template+complement"

        print("%s %s Best location in bin %d,%d zscore = %f,%f" %
              (os.path.splitext(infile)[0], strand, posns[bestidx],
               posns[bestidx], zscore, scores[bestidx]))

        if not args.plot == "stats-only":
            matplotlib.rcParams['font.size'] = 8
            matplotlib.pylab.title(infile+": "+strand)
            matplotlib.pylab.plot(posns[bestidx], scores[bestidx], 'ro')
            if len(scores) < len(posns):
                matplotlib.pylab.plot(posns[1:-1], scores, '.')
            else:
                matplotlib.pylab.plot(posns, scores, '.')
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
    times = times - times[0]
    readlen = len(read)

    if emrescale:
        scaledreads = []
        for index in indexes:
            if index is None:
                continue
            model = index.model
            shift, scale, drift, _, _ = em_rescale.rescale(read, times,
                                                           model.means(), model.sds(), 
                                                           model.kmers())
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

def extend_matches(mappings, binsize, dim,
                   nextend=3, nskip=1,
                   skip_prob=0.2, stay_prob=0.1):
    """
    For each match given, try to extend nextend times, with nskip allowed
    skips or stays
    """
    contributions = mapping_scores(mappings)
    move_prob = 1.-skip_prob-stay_prob

    lookup = collections.defaultdict(lambda: collections.defaultdict(float))
    for readpos, refpos, score in zip(mappings.read_locs, mappings.idx_locs, contributions):
        lookup[readpos][refpos] = score

    def findextension(readpos, refpos, nextend, nskip):
        """
        Given the mappings and scores (in the dict-of-dicts lookup),
        a starting mapping (readpos->refpos), the number to extend,
        and the number of skips allowed, recursively extend the seed
        with the best choice.
        """
        val = lookup[readpos][refpos]
        if val == 0. or nextend == 0:
            return val
        delta_ref = +1 if refpos > 0 else -1
        delta_read = +1 if readpos > 0 else -1
        move = findextension(readpos+delta_read*1, refpos+delta_ref*1, nextend-1, nskip)*move_prob
        skip = 0 if nskip == 0 else findextension(readpos+delta_read*(dim-1), refpos+delta_ref*dim, nextend-1, nskip-1)*skip_prob
        stay = 0 if nskip == 0 else findextension(readpos+delta_read*dim, refpos+delta_ref*(dim-1), nextend-1, nskip-1)*stay_prob
        best = max([move, skip, stay])
        return min(best, val)

    extendedscores = [(start, readpos, refpos, findextension(readpos, refpos, nextend, nskip))
                      for readpos, refpos, start in zip(mappings.read_locs, mappings.idx_locs, mappings.starts)]
    extendedscores = [x for x in extendedscores if x[3] > 0.]

    if len(extendedscores) == 0:
        print("Warning - could not generate sufficiently large extensions")
        return mappings.starts, mappings.read_locs, mappings.idx_locs, contributions
    else:
        return zip(*extendedscores)

    return

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

    starts, _, _, scores = extend_matches(mappings, binsize, dim, nextend, nskip, skip_prob, stay_prob)
    contributions = mapping_scores(mappings)
    move_prob = 1.-skip_prob-stay_prob

    lookup = collections.defaultdict(lambda: collections.defaultdict(float))

    for readpos, refpos, score in zip(mappings.read_locs, mappings.idx_locs, contributions):
        lookup[readpos][refpos] = score

    def findextension(readpos, refpos, nextend, nskip):
        """
        Given the mappings and scores (in the dict-of-dicts lookup),
        a starting mapping (readpos->refpos), the number to extend,
        and the number of skips allowed, recursively extend the seed
        with the best choice.
        """
        val = lookup[readpos][refpos]
        if val == 0. or nextend == 0:
            return val
        delta_ref = +1 if refpos > 0 else -1
        delta_read = +1 if readpos > 0 else -1
        move = findextension(readpos+delta_read*1, refpos+delta_ref*1, nextend-1, nskip)*move_prob
        skip = 0 if nskip == 0 else findextension(readpos+delta_read*(dim-1), refpos+delta_ref*dim, nextend-1, nskip-1)*skip_prob
        stay = 0 if nskip == 0 else findextension(readpos+delta_read*dim, refpos+delta_ref*(dim-1), nextend-1, nskip-1)*stay_prob
        best = max([move, skip, stay])
        return min(best, val)

    extendedscores = [(start, findextension(readpos, refpos, nextend, nskip))
                      for readpos, refpos, start in zip(mappings.read_locs, mappings.idx_locs, mappings.starts)]
    extendedscores = [x for x in extendedscores if x[1] > 0.]

    if len(extendedscores) == 0:
        starts = mappings.starts
        print("Warning - could not generate sufficiently large extensions")
    else:
        starts, contributions = zip(*extendedscores)

    scorebins = numpy.arange(rangemin, rangemax+binsize-1, binsize)
    scores, _ = numpy.histogram(starts, bins=scorebins, weights=scores)

    if return_bins:
        return scores, scorebins
    else:
        return scores

def longest_path(neighbours):
    """
    Given a graph with nodes in topological neighbour and in-neighbours
    in a list-of-lists neighbours, find the longest path through the graph
    """
    pathlengths = numpy.zeros(len(neighbours))
    prev = [None]*len(neighbours)
    for i, neighs in enumerate(neighbours):
        lens = [(pathlengths[j]+1, j) for j in neighs]
        if len(lens) > 0:
            pathlengths[i], prev[i] = max(lens)

    best = numpy.argmax(pathlengths)
    result = [best]
    while prev[best] is not None:
        result.append(prev[best])
        best = prev[best]

    return result[::-1]

def colinear_extended_matches(mappings, binsize, dim,
                               nextend=3, nskip=1,
                               skip_prob=0.2, stay_prob=0.1):
    """
    Returns a bin of scores representing a liklihood of the read beginning
    within that bin of reference positions.
    """
    if mappings is None:
        return [0.], [0]

    starts, readpos, refpos, scores = extend_matches(mappings, binsize, dim, nextend, nskip, skip_prob, stay_prob)
    matches = list(zip(readpos, refpos))

    idxs = [i[0] for i in sorted(enumerate(matches), key=lambda x: x[1])]
    scores = [scores[i] for i in idxs]
    starts = [starts[i] for i in idxs]
    matches = [matches[i] for i in idxs]

    maxdist = 1000

    def valid_neighbour(dest, src):
        """
        Check to see if you can get from dest to src in a way
        that is (a) colinear, and (b) not too far a jump
        """
        if (src[0] >= dest[0]) or (dest[0]-src[0] > maxdist):
            return False
        direction = +1 if dest[1] > 0 else -1
        refdist = (dest[1]-src[1])*direction
        if refdist <= 0 or refdist > maxdist:
            return False
        return True

    neighbours = [[]]*len(matches)
    outneighbours = [[]]*len(matches)
    for i, match in enumerate(matches):
        refstart = match[1] - maxdist*(-1 if match[1] < 0 else 1)
        start = bisect_left(matches, (match[0]-maxdist, refstart))
        start = min(max(0, start), i)
        neighbours[i] = [j for j in range(start, i) if valid_neighbour(matches[i], matches[j])]
        for out in neighbours[i]:
            outneighbours[out].append(i)

    totscores = []
    totmatches = []
    while len(matches) > 0:
        lis = longest_path(neighbours)
        if len(lis) < 3:
            break
        totscores.append(sum([scores[l] for l in lis]))
        totmatches.append(min([starts[l] for l in lis]))

        # remove these matches from the graph
        for matchidx in lis:
            neighbours[matchidx] = []
            for outneigh in outneighbours[matchidx]:
                neighbours[outneigh] = [n for n in neighbours[outneigh] if n != matchidx]

    return totscores, totmatches


if __name__ == "__main__":
    main()
    #cProfile.run('main()','main.profile')
