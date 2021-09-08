import os
from enum import Enum


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def mul_all(x):
    out = 1
    for v in x:
        out *= v
    return out


def get_names(directory):
    names = []
    for path in os.listdir(directory):
        names.append(os.path.join(directory, path))

    return names

class Request(Enum):
    process = 0
    exception = 1
