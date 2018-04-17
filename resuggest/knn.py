#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import sys

import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
# from tokens import EOT, EOU


def get_neighbors(X, Y):
    """
    X coordinates of cluster centers: array,  [n_clusters, n_features]
    Y coordinates of all points: array, [n_samples_a, n_features]
    Retrieves nearest neighbor of Y in X.
    #>>> X = np.array([[0,0,0],[1,1,1]])
    #>>> y = np.array([[0,1,0]])
    #>>> get_neighbors(X, y)
    #[0]
    """
    center_indices = []
    print("Computing medoids")
    for x in X:
        min = np.argmin(pairwise_distances(Y, x.reshape(1,-1)))
        center_indices.append(min)


    center_indices_ = np.array(center_indices)

    # center_indices [n_clusters]
    return center_indices


class KNNAgent():
    def __init__(self,
                 n_neighbors=5,
                 ngram_range=(1, 1),
                 max_features=None,
                 cluster=None,
                 n_jobs=8,
                 ):
        self.vect = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features)
        self.knn = KNeighborsClassifier(
            n_neighbors,
            weights='distance',
            algorithm='brute',
            metric='cosine',
            n_jobs=n_jobs)
        if cluster:
            self.clustering = KMeans(n_clusters=cluster, n_jobs=n_jobs)
        else:
            self.clustering = None
            self.enc = LabelEncoder()

    def __str__(self):
        return "k ={:d}NN Agent {}-grams".format(self.knn.n_neighbors, self.vect.ngram_range)

    def get_params(self):
        """Get parameters for this estimator.
        -------
        params : plain flushed dict of parameters
        """
        return self.__dict__

    def fit(self, X, y):
        """
        X : contexts
        y : utterances
        >>> x = 3
        >>> x == 3
        True
        """
        print("classifier.fit: X.shape", X.shape,"y.shape", y.shape)

        # learns vectorizer on utterances and answers
        self.vect.fit(np.append(X, y))
        # transforms the inputs to vectorized form
        context_vectors = self.vect.transform(X)
        answers = self.vect.transform(y)
        if self.clustering:
            # Fit the clustering on answers
            print("Fitting clustering")
            self.clustering.fit(answers)

            # Get medoids of each cluster
            medoid_indices = get_neighbors(self.clustering.cluster_centers_, answers)
            targets = self.clustering.labels_

            self.cluster_answers = y[medoid_indices]

            # targets : sample -> target
            #print('cluster centers', self.clustering.cluster_centers_.shape)
            #print('cluster answers shape', self.cluster_answers.shape)
        else:
            targets = self.enc.fit_transform(y)
            #print('n_classes', self.enc.classes_.shape[0])

        print("Fitting KNN")
        self.knn.fit(context_vectors, targets)
        print('n_samples', X.shape[0],'\r\nvocabulary size', len(self.vect.vocabulary_),'\r\ntargets', targets.shape)

    def predict(self, question):
        vector_question = self.vect.transform([question])
        p = self.knn.predict_proba(vector_question)[0]  # only one question
        # [p_0, p_1, p_2]
        labels = p.nonzero()[0]
        ind = np.argsort(p[labels])[::-1]
        # use cluster_ind -> medoid mapping
        # inverse transform medoid

        if self.clustering:
            return (self.cluster_answers[labels[ind]], p[labels[ind]])
        else:
            return (self.enc.inverse_transform(labels[ind]), p[labels[ind]])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from utils.conversation import input_loop
    from utils.datasets import load_parallel_text
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        default='tmp/toy/sources.txt',
        help='sources file')
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        default='tmp/toy/targets.txt',
        help='targets file')
    parser.add_argument(
        '-n',
        '--neighbors',
        type=int,
        default=5,
        help='number of neighbors to be considered by KNN')
    #TODO: option for saving clustering and model in pickl to make it faster when just output is needed
    # parser.add_argument(
    #     '-g',
    #     '--n_gram',
    #     type=str,
    #     default="(1,2)",
    #     help='number of neighbors to be considered by KNN')

    args = parser.parse_args()

    print(args.sources,args.targets)

    dframe = load_parallel_text(context_path =args.sources, utterance_path=args.targets)


    KNN = KNNAgent(n_neighbors = args.neighbors, ngram_range =(1, 2), max_features=1000, cluster=100)

    KNN.fit(dframe.Context.values, dframe.Utterance.values)
    input_loop(KNN)
