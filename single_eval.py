#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
Evaluates a single response suggestion agent on fixed fold of data
"""
import sys
import argparse
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from nltk.translate import bleu_score
from resuggest import RetrievalAgent, NonlinearAlignmentAgent, KNNAgent, MLPAgent, RNNAgent, TripletAgent

def load_embedding(for_lang):
    """ Loads an embedding for language. Paths are hard-coded. """
    from gensim.models.keyedvectors import KeyedVectors
    w2v_path, w2v_is_binary = {
        'de': ("/data21/lgalke/vectors/cc.de.300.vec", False),
        'en': ("/data21/lgalke/vectors/GoogleNews-vectors-negative300.bin.gz",
               True)
    }[for_lang]
    print("Loading embedding:", w2v_path)
    embedding = KeyedVectors.load_word2vec_format(w2v_path,
                                                  binary=w2v_is_binary)
    return embedding


def main():
    """ Evaluates all the models """
    parser = argparse.ArgumentParser()

    parser.add_argument('train_sources', type=argparse.FileType('r'),
                        help="Training sources")
    parser.add_argument('train_targets', type=argparse.FileType('r'),
                        help="Training targets")
    parser.add_argument('test_sources', type=argparse.FileType('r'),
                        help="Test sources")
    parser.add_argument('test_targets', type=argparse.FileType('r'),
                        help="Test targets")
    parser.add_argument('-m', '--model', type=str,
                        choices=['knn', 'mlp', 'align', 'retrieval',
                                 'retrieval_gridsearch', 'align_gridsearch',
                                 'rnn', 'triplet'],
                        help="The model to evaluate", default='knn')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to store the results')
    parser.add_argument('-e', '--embed', choices=['en', 'de'], default=None,
                        type=str,
                        help='Language to specify the word embedding to use')
    parser.add_argument('-p', '--predictions', type=argparse.FileType('w'),
                        help='Path to store the predictions', default=None)
    parser.add_argument('-k', type=int, help="Number of neighbors for KNN")
    parser.add_argument('--embedding', type=int,
                        help="Embedding dimension (only used for RNN)",
                        default=100)
    parser.add_argument('--hidden', type=int,
                        help="Number of hidden units per layer.", default=100)
    parser.add_argument('--epochs', type=int, help="Number of epochs.",
                        default=10)
    parser.add_argument('--variant', type=str, default='joint',
                        choices=["joint", "dotproduct", "bilinear"],
                        help="Pick variant for representation alignment")
    parser.add_argument('--similarity', type=str, default="cosine",
                        choices=["cosine", "inner", "mlp",
                                 "bilinear"],
                        help="Pick similarity for rnn")
    parser.add_argument('--ngram-max', type=int, default=1,
                        help="select max ngram range, min ngram range is always 1")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('-n', '--neg', type=int, default=1,
                        help="Negative sampling factor (default 1)")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Be verbose.')
    parser.add_argument('-I', '--no-idf', action='store_false', default=True,
                        help="Don't use IDF.", dest='use_idf')
    parser.add_argument('-M', '--no-matching', action='store_false',
                        default=True, help="Don't use IDF.",
                        dest='use_matching')
    args = parser.parse_args()

    assert args.ngram_max > 0 and args.ngram_max < 5

    print("Loading training data", args.train_sources, args.train_targets)
    train_sources = np.array([l.strip() for l in args.train_sources])
    train_targets = np.array([l.strip() for l in args.train_targets])
    assert len(train_sources) == len(train_targets)
    print("Using", len(train_sources), " pairs for training.")

    print("Loading test data", args.test_sources, args.test_targets)
    test_sources = np.array([l.strip() for l in args.test_sources])
    test_targets = np.array([l.strip() for l in args.test_targets])
    assert len(test_sources) == len(test_targets)
    print("Using", len(test_targets), "pairs for testing.")

    # Partially apply function to not repeat all the time
    def EVAL(model):
        return evaluate(model, train_sources, train_targets,
                        test_sources, test_targets,
                        verbose=args.verbose,
                        store_predictions=args.predictions,
                        store_scores=args.output)

    if args.model == 'retrieval_gridsearch':
        embedding = load_embedding(args.embed)
        for use_matching, use_idf, vectors in itertools.product([True, False],
                                                                [True, False],
                                                                [embedding,
                                                                 None]):
            model = RetrievalAgent(use_matching=use_matching, use_idf=use_idf,
                                   vectors=vectors, lookup_context=True,
                                   ngram_range=(1, args.ngram_max))
            print(model)
            evaluate(model,
                     train_sources,
                     train_targets,
                     test_sources,
                     test_targets,
                     verbose=args.verbose,
                     store_predictions=args.predictions,
                     store_scores=args.output)
    elif args.model == 'align_gridsearch':
        for n_negatives, variant in itertools.product([1,2,3,5,10],
                                                      ['dotproduct', 'joint']):
            agent = NonlinearAlignmentAgent(variant=variant,
                                            n_epochs=args.epochs, 
                                            n_hidden=args.hidden,
                                            negative=n_negatives,
                                            ngram_range=(1, args.ngram_max),
                                            lr=args.lr,
                                            vectors=None)
            EVAL(agent)

    else:
        if args.embed is not None:
            vectors = load_embedding(args.embed)
        else:
            vectors = None
        print("Selecting model:", args.model)
        model = {
            'retrieval': RetrievalAgent(use_matching=args.use_matching,
                                        vectors=vectors,
                                        use_idf=args.use_idf,
                                        lookup_context=True),
            'align': NonlinearAlignmentAgent(variant=args.variant,
                                             n_epochs=args.epochs,
                                             n_hidden=args.hidden,
                                             negative=args.neg,
                                             lr=args.lr,
                                             ngram_range=(1, args.ngram_max),
                                             vectors=vectors),
            'knn': KNNAgent(n_neighbors=args.k,
                            ngram_range=(1, args.ngram_max)),
            'mlp': MLPAgent(hidden_layer_sizes=(args.hidden,),
                            ngram_range=(1, args.ngram_max)),
            'rnn': RNNAgent(hidden_size=args.hidden,
                            n_epochs=args.epochs,
                            lr=args.lr,
                            embedding_size=args.embedding, negative=args.neg,
                            share_vocab=True,
                            similarity=args.similarity),
            'triplet': TripletAgent(n_hidden=args.hidden, n_epochs=args.epochs,
                                    lr=args.lr, p=2, n_negative=args.neg)

        }[args.model]
        print(model)
        evaluate(model,
                 train_sources,
                 train_targets,
                 test_sources,
                 test_targets,
                 verbose=args.verbose,
                 store_predictions=args.predictions,
                 store_scores=args.output)


def mp(targets, responses, n_bleu=1):
    """
    Computes modified precision scores for a bunch of response-target pairs
    """
    return np.array([float(bleu_score.modified_precision(targets[i],
                                                         responses[i],
                                                         n=n_bleu))
                     for i in range(len(targets))])


def evaluate(model,
             train_sources,
             train_targets,
             test_sources,
             test_targets,
             verbose=False,
             store_predictions=None,
             store_scores=None):
    """ Evaluates a response suggestion agent """
    print("Training...")
    model.fit(train_sources, train_targets)
    print("Training succeeded.")

    print("Suggesting Responses for test data")
    responses = []
    for question in test_sources:
        suggestions, __scores = model.predict(question)
        try:
            response = suggestions[0]
        except IndexError:
            response = '__NO_RESPONSE__'
        if verbose:
            print()
            print(question, '<>', response, sep='\n')
            print()
        responses.append(response)
        if store_predictions:
            print(response, file=store_predictions)

    tokenize = CountVectorizer().build_analyzer()

    print("Splitting utterances into words")
    responses = [tokenize(r) for r in responses]
    # and creating lists to prepare for BLEU score
    targets = [[tokenize(t)] for t in test_targets]
    smooth = bleu_score.SmoothingFunction().method5
    print("Evaluating suggested responses (Smoothing method 5)")
    scores = {
        'mp1': mp(targets, responses, n_bleu=1),
        'mp2': mp(targets, responses, n_bleu=2),
        'mp3': mp(targets, responses, n_bleu=3),
        'bleu':  bleu_score.corpus_bleu(targets, responses,
                                        smoothing_function=smooth)
    }
    log(model, scores, path=store_scores)
    return scores


def log(model, scores, path=None):
    """ Logs model description along with evaluated scores """
    if path is None:
        print('-' * 79)
        print(model)
        log_scores(scores)
        print('-' * 79)
    else:
        with open(path, 'a') as fhandle:
            print('-' * 79, file=fhandle)
            print(model, file=fhandle)
            log_scores(scores, file=fhandle)
            print('-' * 79, file=fhandle)


def log_scores(scores, file=sys.stdout):
    """ Logs the scores either to sys.stdout or file """
    mp1, mp2, mp3, bleu = scores['mp1'], scores['mp2'], scores['mp3'], scores['bleu']
    print("MP1: {:.4f}, ({:.2f})".format(mp1.mean(), mp1.std()), file=file)
    print("MP2: {:.4f}, ({:.2f})".format(mp2.mean(), mp2.std()), file=file)
    print("MP3: {:.4f}, ({:.2f})".format(mp3.mean(), mp3.std()), file=file)
    print("BLEU: {:.4f}".format(bleu), file=file)


if __name__ == '__main__':
    main()
    print("Done.")
