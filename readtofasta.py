#!/usr/bin/env python
"""
Simple fast5s-to-fasta extractor
"""
from __future__ import print_function
import argparse
import h5py
import sys
import os

def readfasta(filename, calls="2D"):
    """
    Reads the fastq from basecalled read, returns label and sequence
    """
    with h5py.File(filename, "r") as fast5_file:
        calls_group = "BaseCalled_"+calls
        eventpath = "/Analyses/Basecall_2D_000/"+calls_group
        if not eventpath in fast5_file:
            eventpath = "/Analyses/Basecall_1D_000/"+calls_group

        if not "Analyses" in fast5_file or not "Basecall_2D_000" in fast5_file["Analyses"] or \
           not calls_group in fast5_file["Analyses"]["Basecall_2D_000"]:
            raise KeyError("Not present in HDF5 file")
        if not "Fastq" in fast5_file[eventpath]:
            raise KeyError("Fastq not found in HDF5 file")
        data = fast5_file["Analyses"]["Basecall_2D_000"][calls_group]["Fastq"]

        lines = data[()].splitlines()

        seq = lines[1].decode('utf-8')

        label = lines[0][1:].strip()
        label_suffix = "_twodirections" if calls == "2D" else calls
        label = label+label_suffix
        label = label.decode('utf-8')

    return label, seq

def build_filelist(filenames):
    """ If given a directory, descend into directory and generate
        list of files (but do not descend into subdirectories)."""
    file_list = []
    for filename in filenames:
        if os.path.isfile(filename):
            file_list.append(filename)
        elif os.path.isdir(filename):
            for dirfile in os.listdir(filename):
                pathname = os.path.join(filename, dirfile)
                if os.path.isfile(pathname) and pathname.endswith('.fast5'):
                    file_list.append(pathname)
    return file_list


def main():
    " Driver program "
    parser = argparse.ArgumentParser(description="Extact basecalled (2D or 1D) FASTAs")
    parser.add_argument('infile', nargs="+", type=str)
    parser.add_argument('-l', '--linelength', type=int, default=60)
    parser.add_argument('-c', '--calls', choices=["template", "complement", "2D"],
                        default="2D")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help="Specify output file (default:stdout)")

    args = parser.parse_args()

    files = build_filelist(args.infile)
    seqs = []
    for infile in files:
        try:
            label, sequence = readfasta(infile)
            seqs.append((label+" "+infile, sequence))
        except:
            continue

    outf = args.output
    llen = args.linelength

    for label, sequence in seqs:
        print(">"+label, file=outf)
        nchunks = (len(sequence)+llen-1)//llen
        for i in range(nchunks):
            lstart = i*llen
            lend = min(lstart + llen, len(sequence))
            print(sequence[lstart:lend], file=outf)
    return 0

if __name__ == "__main__":
    sys.exit(main())
