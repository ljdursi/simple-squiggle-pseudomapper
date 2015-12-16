#!/usr/bin/env python
"""
Contains PoreModel class, along with test driver

 Jonathan Dursi, Jonathan.Dursi@oicr.on.ca
 SimpsonLab, OICR
"""
from __future__ import print_function
import argparse
import csv
import sys
import os
import numpy

HAVE_HDF5 = False
try:
    import h5py
    HAVE_HDF5 = True
except ImportError:
    pass

class PoreModel(object):
    """
    The pore model class

    A pore model is, for a given size k, a mapping from kmers
    to expected signal levels and standard deviations.

    The PoreModel class contains initialization methods for
    reading in the model from CSV files or from Fast5 files,
    and then converting a genomic sequence to a series of
    levels and std deviations.

    It also contains a method for scaling a read to a given
    pore model, so that first two moments of the read levels
    the average and the std deviation) match that of the
    model; this works reasonably well for long-enough and
    representative-enough reads.
    """
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def __from_fast5(cls, filename, complement=False):
        """
        Import a pore model from an ONT Fast5 file
        """
        fast5 = h5py.File(filename, "r")
        calls = "BaseCalled_"
        if complement:
            calltype = "complement"
        else:
            calltype = "template"
        path = "/Analyses/Basecall_2D_000/"+calls+calltype+"/Model"
        if not path in fast5:
            path = "/Analyses/Basecall_1D_000/"+calls+calltype+"/Model"
        if not path in fast5:
            raise KeyError("Not found: "+path)
        model_table = fast5[path]

        level_mean = {}
        level_stdv = {}
        for row in model_table:
            kmer = row["kmer"]
            level_mean[kmer] = numpy.float32(row["level_mean"])
            level_stdv[kmer] = numpy.float32(row["level_stdv"])

        groupname = "/Analyses/Basecall_2D_000/Summary/basecall_1d_"
        modelfilename = fast5[groupname+calltype].attrs["model_file"]
        return level_mean, level_stdv, modelfilename, model_table.attrs["shift"], \
                model_table.attrs["scale"]

    @classmethod
    def __from_csv(cls, filename, delimiter=","):
        """
        Import a pore model from a CSV file
        CSV file must have headers "kmer", "level_mean", and "level_stdv"
        """
        level_mean = {}
        level_stdv = {}

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            for row in reader:
                kmer = row["kmer"]
                level_mean[kmer] = numpy.float32(row["level_mean"])
                level_stdv[kmer] = numpy.float32(row["level_stdv"])

        return level_mean, level_stdv

    def __init__(self, filename, complement=False, delimiter=","):
        _, extension = os.path.splitext(filename)

        self.__filename = filename
        if extension in [".fast5", ".f5", ".h5", ".hdf5"]:
            if not HAVE_HDF5:
                raise ImportError("Could not load module h5py")
            self.__level_mean, self.__level_stdv, name, _, _ = self.__from_fast5(filename, complement)
            self.__name = name
        else:
            self.__level_mean, self.__level_stdv = self.__from_csv(filename, delimiter)
            self.__name = filename

        any_kmer = list(self.__level_mean.keys())[0]
        self.__k = len(any_kmer)

        means = list(self.__level_mean.values())
        stds = list(self.__level_stdv.values())

        self.__meanavg = numpy.mean(means)
        self.__meanstd = numpy.std(means)
        self.__stdvavg = numpy.mean(stds)
        self.__stdvstd = numpy.std(stds)


    def to_csv(self, csvfile, delimiter=","):
        """
        Outputs model as CSV file - presumably after being read in from FAST5 file
        """
        fieldnames = ['kmer', 'level_mean', 'level_stdv']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)

        writer.writeheader()
        for kmer in self.__level_mean:
            writer.writerow({'kmer':kmer, 'level_mean':self.__level_mean[kmer],
                             'level_stdv':self.__level_stdv[kmer]})

    def sequence_to_events(self, sequence):
        """
        Returns a series of events given the pore model and the input sequence.

        Returns; events (1d numpy array - level means)
                 stds   (1d numpy array - level std devs)
        """
        seqlen = len(sequence)
        nevents = seqlen - self.__k + 1
        if nevents <= 0:
            return numpy.array([]), numpy.array([])

        events = numpy.zeros(nevents)
        sds = numpy.zeros(nevents)

        for i in range(0, nevents):
            kmer = sequence[i:i+self.__k]
            try:
                events[i], sds[i] = self.__level_mean[kmer], self.__level_stdv[kmer]
            except KeyError:
                events[i], sds[i] = -1, -1

        return events, sds

    @property
    def k(self):
        """Returns the pore model's k"""
        return self.__k

    @property
    def name(self):
        """Returns the pore model name"""
        return self.__name

    def scale_events(self, events, sds=None):
        """
        Scales the input events, and optionally input std deviations,
        according to the model - so that the events have the same mean
        and std deviation as the level means in the model.

        Returns: events (1d numpy array, scaled events)
                 stdvs  (1d numpy array, scaled std deviations, if sds != None)
        """
        events_scale = self.__meanstd/numpy.std(events)
        events_shift = self.__meanavg - events_scale*numpy.mean(events)

        if not sds is None:
            sds_scale = self.__stdvstd/numpy.std(sds)
            sds_shift = self.__stdvavg - sds_scale*numpy.mean(sds)

            return events*events_scale+events_shift, sds*sds_scale+sds_shift

        return events*events_scale + events_shift

    def means(self):
        """Returns the level means of the model as an array"""
        return numpy.array(list(self.__level_mean.values()))

    def sds(self):
        """Returns the level sds of the model as an array"""
        return numpy.array(list(self.__level_stdv.values()))

    def kmers(self):
        """Returns kmers in the model in the same order as sds & means"""
        return numpy.array(list(self.__level_mean.keys()))


def extractmodel():
    """ simple driver program - extract a model from a file, test it against a sequence """
    parser = argparse.ArgumentParser()

    test_sequence = "ACGTACGTACGT"
    parser.add_argument('infilename', type=str, help="Input file to extract model from")
    parser.add_argument('outfile', nargs="?", type=argparse.FileType('w'),
                        default=sys.stdout, help="Output CSV file (default: stdout)")
    parser.add_argument('-C', '--complement', action="store_true",
                        help="Complement model (default: template model)")
    parser.add_argument('-D', '--delimiter', type=str, default=",",
                        help="Delimiter to use if CSV file")
    parser.add_argument('-s', '--sequence', type=str, default=test_sequence,
                        help="Sequence to test against model")
    args = parser.parse_args()

    model = PoreModel(args.infilename, args.complement, args.delimiter)
    print("model name: "+model.name, file=sys.stderr)
    print(args.sequence+" -> "+str(model.sequence_to_events(args.sequence)[0]),
          file=sys.stderr)
    model.to_csv(args.outfile, args.delimiter)

if __name__ == "__main__":
    extractmodel()
