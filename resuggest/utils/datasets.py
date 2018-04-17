# /usr/bin/env python3
# -*- coding=utf-8 -*-

import hashlib
import os
import random

import numpy as np

import pandas as pd


def sample_distractors(context, utterance, n_distractors=9):
    """ Samples distractors (unused)"""
    distractors = []
    utterance = list(utterance)

    for i in range(len(context)):
        distractors += [
            random.sample(utterance[:i] + utterance[(i + 1):], n_distractors)
        ]
    distractors = np.array(distractors).transpose()

    return distractors


def distracted_dataframe(context, utterance, distract=0):
    """ Enriches a dataframe with distractions """
    d = {'Context': context, 'Utterance': utterance}

    if distract:
        distractors = sample_distractors(context, utterance, distract)

        for i, dis in enumerate(distractors):
            d['Distractor_%d' % i] = dis

    return pd.DataFrame(d)


def load_parallel_text(context_path, utterance_path, distract=0):
    """ Cached reading of parallel text format, optionally distracted """
    m = hashlib.sha1()

    # Cached Read
    for somepath in [context_path, utterance_path]:
        m.update(os.path.abspath(somepath).encode('utf-8'))
    m.update(str(distract).encode('utf-8'))

    os.makedirs('tmp', exist_ok=True)
    path = 'tmp/' + m.hexdigest() + '.pkl'
    try:
        df = pd.read_pickle(path)
        print("Reading", path)
    except FileNotFoundError:
        with open(context_path, 'r') as f:
            context = np.array([line.strip() for line in f.readlines()])
        with open(utterance_path, 'r') as f:
            utterance = np.array([line.strip() for line in f.readlines()])
        df = distracted_dataframe(context, utterance, distract=distract)
        print("Writing", path)
        df.to_pickle(path)

    return df
