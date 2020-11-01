import csv
import os

import numpy as np


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    file = os.path.join(input_file,'test.tsv')
    with open(file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def _create_sessionids(data_dir):
    """Creates examples for the training and dev sets."""
    lines=_read_tsv(data_dir)
    sessionids = []
    for line in lines:
        sessionids.append(line[0])
    return sessionids


def select_sessionids_index(data_dir):
    sessionids = _create_sessionids(data_dir)
    temp = 0
    indexList = []
    for i in range(0, len(sessionids) - 1):
        if (sessionids[i + 1] != sessionids[temp]):
            indexList.append(temp)
        temp = i + 1
    indexList.append(len(sessionids) - 1)
    return indexList


def session_probability(data_dir, preds, labelIndex=0):
    indexList = select_sessionids_index(data_dir)
    probability = []
    point = 0
    i = 0
    while i < len(indexList):
        location = indexList[i]
        avg = np.sum(preds[point:location + 1], axis=0) / (location - point + 1)
        # softMax = np.exp(avg) / sum(np.exp(avg))
        probability.append(avg[labelIndex])
        point = location + 1
        i += 1
    return probability

# session-role-text
def session_probability(data_dir, preds, labelIndex=0):
    indexList = select_sessionids_index(data_dir)
    probability = []
    point = 0
    i = 0
    while i < len(indexList):
        location = indexList[i]
        avg = np.sum(preds[point:location + 1], axis=0) / (location - point + 1)
        softMax = np.exp(avg) / sum(np.exp(avg))
        probability.append(softMax[labelIndex])
        point = location + 1
        i += 1
    return probability
