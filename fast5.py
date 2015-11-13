"""
Simple HDF5 Fast5 accessor routines
"""
import h5py
import numpy

def read_basecalled_events(filename, complement=False, include_stds=False):
    """
    Read basecalled events from FAST5 file
    Returns numpy array of events levels, and optionally of std deviations
    """
    events = []
    sds = []
    fast5 = h5py.File(filename, "r")
    calls = "BaseCalled_"
    if complement:
        calls += "complement"
    else:
        calls += "template"
    path = "/Analyses/Basecall_2D_000/"+calls+"/Events"
    if not path in fast5:
        path = "/Analyses/Basecall_1D_000/"+calls+"/Events"
    if not path in fast5:
        raise KeyError("Not found: "+path)
    data = fast5[path]
    for row in data:
        events.append(row[0])
        sds.append(row[2])

    if include_stds:
        return (numpy.array(events, dtype=numpy.float32), numpy.array(sds, dtype=numpy.float32))
    else:
        return numpy.array(events, dtype=numpy.float32)

def read_raw_events(filename, complement=False, include_stds=False):
    """
    Read raw events from FAST5 file
    Returns numpy array of events levels, and optionally of std deviations
    """
    events = []
    sds = []
    fast5 = h5py.File(filename, "r")

    if (not "Analyses" in fast5 or not "EventDetection_000" in fast5["Analyses"] or
            not "Reads" in fast5["Analyses"]["EventDetection_000"]):
        raise KeyError("Not present in HDF5 file")

    reads = []
    for read in fast5["/Analyses/EventDetection_000/Reads"]:
        reads.append(read)

    attrs = fast5["/Analyses/EventDetection_000/Reads"][reads[0]].attrs
    hairpin_index = None
    if attrs["hairpin_found"] == 1:
        hairpin_index = attrs["hairpin_event_index"]

    data = fast5["/Analyses/EventDetection_000/Reads"][reads[0]]["Events"]
    for row in data:
        events.append(row["mean"])
        sds.append(row["variance"])

    events = numpy.array(events, dtype=numpy.float32)
    sds = numpy.array(sds, dtype=numpy.float32)

    events = events[50:-30]
    sds = sds[50:-30]

    if complement and hairpin_index is None:
        raise KeyError("Asking for Complement Events and Hairpin Not Found")

    if not complement and hairpin_index is not None:
        events = events[:hairpin_index-30]
        sds = sds[:hairpin_index-30]

    if complement and hairpin_index is not None:
        events = events[hairpin_index+80:]
        sds = sds[hairpin_index+80:]

    if include_stds:
        return events, sds
    else:
        return events

def readevents(filename, complement=False, include_stds=False):
    """
    Read events from FAST5 file - template strand by default
    Attempt basecalled events first, and otherwise return raw events
    Returns numpy array of events levels, and optionally of std deviations
    """
    try:
        result = read_basecalled_events(filename, complement, include_stds)
    except KeyError:
        result = read_raw_events(filename, complement, include_stds)
    return result
