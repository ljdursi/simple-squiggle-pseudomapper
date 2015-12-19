#!/usr/bin/env python
"""
Demonstrates EM calculation of shift, scale, drift, and scale_sd parameters
for normalizing model to read.
"""
from __future__ import print_function
import sys
import argparse
import scipy.stats
import scipy.sparse
import numpy
import poremodel
import fast5

def init_scaling_parameters(events, means):
    """
    Estimate initial shift, scale by comparing means, sds of event means
    to level means.
    """
    drift = 0.
    scale_var = 1.5 

    scale = numpy.std(events)/numpy.std(means)
    shift = numpy.mean(events) - numpy.mean(scale*means)

    return shift, scale, drift, scale_var

def event_probabilities(events, times, shift, scale, drift, scale_var, means, sds):
    """ As the E-step, crudely assign a probability for event i to
        correspond to state j using the observed mean and the
        level means and sds."""
    nevents = events.size
    nlevels = means.size

    sevents = events - drift*times
    smeans = means*scale + shift
    ssds = sds * scale_var

    pij = numpy.zeros((nevents, nlevels), dtype=numpy.float32)
    for i, sevent in enumerate(sevents):
        pij[i, :] = scipy.stats.norm.pdf(sevent, smeans, ssds)
        pij[i, :] = pij[i, :]/numpy.sum(pij[i, :])

    return pij

def update_scale_parameters(pij, shift, scale, drift, scale_var, oldweight, 
                            events, times, means, sds):
    """
    Update the scaling parameters given a posteriori event
    assignment probabilities
    """
    nstates = means.size
    wij = pij/(sds*sds)
    A = numpy.zeros((3, 3))

    wij_means = numpy.dot(wij, means)

    A[0, 0] = numpy.sum(wij)
    A[1, 1] = numpy.sum(numpy.dot(wij, means*means))
    A[2, 2] = numpy.sum(numpy.dot(wij.T, times*times))

    A[0, 1] = numpy.sum(wij_means)
    A[1, 0] = A[0, 1]

    A[0, 2] = numpy.sum(numpy.dot(wij.T, times))
    A[2, 0] = A[0, 2]

    A[1, 2] = numpy.dot(times, wij_means)
    A[2, 1] = A[1, 2]

    bvec = numpy.array([numpy.sum(numpy.dot(wij.T, events)),
                        numpy.dot(wij_means, events),
                        numpy.sum(numpy.dot(wij.T, events*times))])

    A    = A    + numpy.diag([oldweight]*3)
    bvec = bvec + numpy.dot(numpy.diag([oldweight]*3), numpy.array([shift, scale, drift]))

    x = numpy.linalg.solve(A, bvec)
    shift, scale, drift = tuple(x)

    y = events - drift*times - shift - scale*numpy.reshape(means, (nstates, 1))
    y = y*y

    scale_var = numpy.sum(wij*y.T)
    scale_var = scale_var / numpy.sum(pij)
    scale_var = numpy.sqrt(scale_var)

    return shift, scale, drift, scale_var

def incorporate_transitions(pij, transition):
    """
    Given initial, purely local, probabilities for read events to be
    associated with model levels, incorporate information from
    the transition matrices to propagate that information further.

    inputs: the probability matrix pij that gives probability of
            read event i -> model level j,
            sparse transition matrices T and T.T
    outputs: updated pij
    """
    sparsetrans, sparsetrans_inv = transition
    pnext = (sparsetrans_inv.dot(pij.T)).T
    pprev = (sparsetrans.dot(pij.T)).T

    pij[:-1, :] = pij[:-1, :] * pprev[1:, :]
    pij[1:, :] = pij[1:, :] * pnext[:-1, :]
    pij = pij/numpy.sum(pij, 1, keepdims=True)
    return pij

def rescale(event_means, event_starts, level_means, level_sds, level_kmers,
            oldweight=0., niters=10, ntransitions=2, verbose=False):
    """
    Perform niters iterations of an EM scheme to rescale the read to a model;
    returns rescaling and a probability matrix assigning events to model kmers.

    inputs:
        event_means, event_starts : means and starts from the read
        level_means, level_sds, level_kmers: means, std devs,
          and kmers from the model
        oldweight: if non zero, underrelax EM step by including
          oldweight fraction of prev step
        niters: number of iterations, default 10
        ntransitions: if non zero, use this many iterations to adjust purely
          gaussian-distribution-based calculations of probabilities with
          transition probabilities
        verbose: report on iterations to stdout if True

    returns:
        shift, scale, drift, scale_var, pij
    """
    shift, scale, drift, scale_var = init_scaling_parameters(event_means,
                                                             level_means)
    if ntransitions > 0:
        transition = build_transition_matrix(level_kmers)

    for i in range(niters):
        if verbose:
            print(i, shift, scale, drift, scale_var)
        pij = event_probabilities(event_means, event_starts, shift, scale,
                                  drift, scale_var, level_means, level_sds)
        if ntransitions > 0:
            # ramp up the number of transitions
            # for both performance and accuracy
            loc_ntransitions = (i*ntransitions)//niters + 1
            for _ in range(loc_ntransitions):
                pij = incorporate_transitions(pij, transition)
        shift, scale, drift, scale_var = \
                update_scale_parameters(pij, shift, scale, drift, scale_var,
                                        oldweight, event_means, event_starts,
                                        level_means, level_sds)

    if verbose:
        print(niters, shift, scale, drift, scale_var)

    return shift, scale, drift, scale_var, pij

def main():
    """
    Main program if run from the command line; run on provided input files
    and report rescaling quantities
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Fast5 file', nargs='+')
    parser.add_argument('--strand', choices=['template', 'complement'],
                        default='template', help='Strand:')
    parser.add_argument('-n', '--niters', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('-d', '--damp', action='store_true',
                        help='Include previous iteration*prevweight')
    parser.add_argument('-p', '--prevweight', type=float, default=0.1,
                        help='Weight previous iteration if damping')
    parser.add_argument('-t', '--transition', action='store_true',
                        help='Use transition probabilities')
    parser.add_argument('-T', '--numtransitions', type=int, default=2,
                        help='Number of transition iterations to use')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print output each iteration')
    parser.add_argument('-m', '--model', type=str,
                        help='Optionally use a provided model')
    args = parser.parse_args()

    oldweight = 0. if not args.damp else args.prevweight
    numtransitions = 0 if not args.transition else args.numtransitions

    for fn in args.input:
        strand = True if args.strand == "complement" else False
        try:
            event_means, event_starts, _, read_kmers = fast5.read_basecalled_events(fn, strand)
            event_starts = event_starts - event_starts[0]
            if not args.model:
                model = poremodel.PoreModel(fn, True if strand == 'complement' else False)
            else:
                model = poremodel.PoreModel(args.model, True if strand == 'complement' else False)
            level_means = model.means()
            level_sds = model.sds()
            kmers = model.kmers()
        except:
            print("Could not access fields in file "+fn)
            continue

        shift, scale, drift, scale_var, pij = \
                rescale(event_means, event_starts, level_means, level_sds,
                        kmers, oldweight, args.niters, numtransitions, args.verbose)

        if args.verbose:
            n_in_top5 = 0
            for i, row in enumerate(pij):
                topidxs = numpy.argsort(row)[-5:]
                topkmers = [kmers[idx] for idx in topidxs]
                if read_kmers[i] in topkmers:
                    n_in_top5 += 1
            print('Frac of metrichor kmers in top 5: '+str(n_in_top5*1.0/len(event_means)))


        print(fn)
        print("Shift:    "+str(shift))
        print("Scale:    "+str(scale))
        print("Drift:    "+str(drift))
        print("Scale_sd: "+str(scale_var))

    return 0


def build_transition_matrix(kmers, skip_prob=0.2, stay_prob=0.1):
    """
    Build a transition matrix T s.t. T_{i,j} = prob(transition from
    event i to event j)

    We're only considering 0-, 1-, and 2-moves.

    inputs: list of kmers in their order in the model,
            skip and stay probabilities (defaults appropriate to SQK006)
    outputs: returns a |kmers|x|kmers| transition matrix
    """
    # build an alphabet, as well as a dictionary going from
    # kmer to index
    alphabet = ""
    kmers_to_idx = {}
    for idx, kmer in enumerate(kmers):
        kmers_to_idx[kmer] = idx
        for base in kmer:
            if not base in alphabet:
                alphabet = alphabet + base

    nalphabet = len(alphabet)
    nkmers = len(kmers)

    move_prob = 1. - skip_prob - stay_prob
    move_prob_per_kmer = move_prob / nalphabet
    skip_prob_per_kmer = move_prob / (nalphabet*nalphabet)
    T = numpy.zeros((nkmers, nkmers), dtype=numpy.float32)

    diag = numpy.arange(nkmers)
    T[diag, diag] = stay_prob

    for idx, kmer in enumerate(kmers):
        one_moves = [kmers_to_idx[kmer[1:]+base] for base in alphabet]
        two_moves = [kmers_to_idx[kmer[2:]+base1+base2]
                     for base1 in alphabet
                     for base2 in alphabet]
        T[idx, one_moves] += move_prob_per_kmer
        T[idx, two_moves] += skip_prob_per_kmer

    return scipy.sparse.csr_matrix(T), scipy.sparse.csr_matrix(T.T)

if __name__ == "__main__":
    sys.exit(main())
