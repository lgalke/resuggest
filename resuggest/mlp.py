import argparse
import numpy as np
from nltk.tokenize import word_tokenize

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder



class MLPAgent():
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation="relu",
                 solver="lbfgs",  # faster and better than adam for small data http://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
                 max_iter=200,
                 verbose=True,
                 early_stopping=False,
                 ngram_range=(1, 1),
                 max_features=None):
        self.vect = TfidfVectorizer(
            tokenizer=word_tokenize,
            ngram_range=ngram_range,
            max_features=max_features)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            #early_stopping=early_stopping,
            verbose=verbose)
        self.enc = LabelEncoder()



    def __str__(self):
        return "{}-layer MLP-Agent {}-grams".format(self.mlp.hidden_layer_sizes,
                                                self.vect.ngram_range)
            #"{:d}-NN Agent {}".format(self.get_params())

    def get_params(self):
        """Get parameters< for this estimator.
        -------
        params : plain flushed dict of parameters
        """

        # dict = {}
        # for object_name, values in self.__dict__:
        #     dict[object_name]


        return self.__dict__

    def fit(self, X, y):
        """
        X : question
        y : utterances
        >>> x = 3
        >>> x == 3
        True
        """
        #X,y = X[:5000], y[:5000]
        print("classifier.fit: X.shape", X.shape,"y.shape", y.shape)

        # learns vectorizer on utterances and answers
        self.vect.fit(np.append(X, y))
        # transforms the inputs to vectorized form
        questions_vec = self.vect.transform(X)
        #answers_vec = self.vect.transform(y)
        #TODO: replace answer values with labels

        answers_vec = self.enc.fit_transform(y)

        self.mlp.fit(questions_vec,answers_vec)
        print("Fitting MLP")

        print('n_samples', X.shape[0])
        print('vocabulary size', len(self.vect.vocabulary_))
        print('targets', answers_vec.shape)

    def predict(self, question):
        vector_question = self.vect.transform([question])
        p = self.mlp.predict_proba(vector_question)[0]  # only one question
        # [p_0, p_1, p_2]
        labels = p.nonzero()[0]
        ind = np.argsort(p[labels])[::-1]
        #print("p ",p," labels ",labels, " ind ",ind)
        # use cluster_ind -> medoid mapping
        # inverse transform medoid
        return (self.enc.inverse_transform(labels[ind]), p[labels[ind]])
        #    return (self.cluster_answers[labels[ind]], p[labels[ind]])


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
        '-af',
        '--activation_function',
        type=str,
        default='relu',
        help='activation function for MLP: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’')
    args = parser.parse_args()

    print(args.sources,args.targets)

    dframe = load_parallel_text(context_path =args.sources, utterance_path=args.targets)


    MLP = MLPAgent(ngram_range =(1, 2), max_features=10000, hidden_layer_sizes= (300,))
    print(dframe.shape)
    MLP.fit(dframe.Context.values, dframe.Utterance.values)
    input_loop(MLP)
