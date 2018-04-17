import argparse
import itertools
import doctest
import numpy as np
from nltk.translate import bleu_score

from utils.datasets import load_parallel_text
from knn import KNNAgent
from mlp import MLPAgent
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils.tokens import EOT, EOU
from sklearn.feature_extraction.text import CountVectorizer


tokenize = CountVectorizer().build_analyzer()


def tokenize_modified_precision(reference, candidate, n=3, verbose=False):
    """

    :param reference:
    :param candidate:
    :param n:
    :return:
    # TODO:
    >>> tokenize_modified_precision("der String von a","der String von b", n=1 )
    0.75
    """
    if verbose: print("#"*30+"\r\ncandidate:\r\n",candidate,"\r\nreference:",reference)
    candidate, reference = tokenize(candidate), tokenize(reference)
    bleu = float(bleu_score.modified_precision([reference], candidate, n=n))

    return bleu


def calc_corpus_BLEU_from_file(prediction_file,test_file):
    #TODO: make several references possible

    # >> > list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    # >> > hypotheses = [hyp1, hyp2]
    # >> > corpus_bleu(list_of_references, hypotheses)  # doctest: +ELLIPS
    hypotheses = []
    with open(prediction_file,'r') as pf:
        for candidate in pf:
            hypotheses.append(tokenize(candidate))

    list_of_references = []
    with open(test_file,'r') as tf:
        for reference in tf:
            list_of_references.append([tokenize(reference)])
    smf = bleu_score.SmoothingFunction().method5
    bscore = bleu_score.corpus_bleu(list_of_references, hypotheses,smoothing_function=smf)
    return bscore



def tokenize_bleu_score(reference, candidate, verbose=False):
    """

    :param reference:
    :param candidate:
    :param n:
    :return:
    # TODO:
    >>> tokenize_bleu_score("der String von a","der String von b", n=1 )
    0.75
    """
    if verbose: print("#"*30+"\r\ncandidate:\r\n",candidate,"\r\nreference:",reference)
    candidate, reference = tokenize(candidate), tokenize(reference)
    bleu = float(bleu_score.sentence_bleu([reference], candidate,smoothing_function=bleu_score.SmoothingFunction().method1))

    return bleu



def single_file_evaluation(sources='tmp/sources.txt', targets='tmp/targets.txt', bleu_n=3):
    time = str(datetime.now())
    filename = "evaluations/eval_{}.txt".format(time).replace(' ', '_').replace(':', '-')
    fhandle = open(filename,"w")
    fhandle.write(time+"\r\n"+"dataset: "+sources+" "+targets)
    fhandle.close()
    print("created file: {}".format(filename))
    for k, ngram_range in itertools.product(range(3, 7), [(1,2)]):
        agent = KNNAgent(n_neighbors = k,ngram_range= ngram_range)
        evalAgent(agent, bleu_n=bleu_n,sources=sources,targets=targets, result_file=filename)
    for hidden_L_size,ngram_range in itertools.product([(100,),(200,),(300,),(400,)],[(1,2)]):
        agent = MLPAgent(hidden_layer_sizes=hidden_L_size, ngram_range=ngram_range,verbose=True)
        evalAgent(agent, bleu_n=bleu_n, sources=sources, targets=targets, result_file=filename)


def evalAgent(agent,bleu_n = 3,sources='tmp/sources.txt',targets='tmp/targets.txt',verbose=False, result_file=None):
    df = load_parallel_text(sources, targets)
    X_train, X_test, y_train, y_test = train_test_split( df.Context.values, df.Utterance.values)
    # X_train = X_test = df.Context.values
    # y_train = y_test = df.Utterance.values
    print(agent," evaluation starts{}".format(str(datetime.now())))
    print("fitting", len(X_train), "samples")
    agent.fit(X_train, y_train)
    print("sucessfully fitted")
    print(agent, "predicting", len(X_test), "samples")
    predictions = []
    for test_history in X_test:
        prediction = agent.predict(test_history)
        if verbose: print("#" * 30 + "\r\ntest_history (input):\r\n", test_history, "\r\n\r\n predicted answer\r\n", prediction[0])
        predictions.append(prediction)
    # pred = [agent.predict(val) for val in X_test]
    print("predictions successfull\r\nretrieving BLEU score")
    print("Removing EOU and EOT tokens...")
    predictions = [(a[0].replace(EOT, '').replace(EOU, ''), p) for (a, p) in
                   predictions]

    results_s = ""
    for n in range(1, bleu_n + 1):
        bleu_scores = np.asarray([
            tokenize_modified_precision(utter, guess, n=n, verbose=verbose)
            for utter, (guess, _) in zip(y_test, predictions)
        ])

        results_s += "BLEU-{:d}: {:.2f} ({:.2f})\r\n".format(n, bleu_scores.mean(), bleu_scores.std())
    results_s = "{0}:\r\n{1}".format(agent, results_s)
    print(results_s)
    if result_file is not None:
        fhandle = open(result_file, "a")
        fhandle.write(results_s)
        fhandle.close()
        print("results saved to {}".format(result_file))
    else:
        print("no such file found. results will not be saved")
    #return results_s



if __name__ == '__main__':
    doctest.testmod()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        default='tmp/sources.txt',
        help='sources file')
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        default='tmp/targets.txt',
        help='targets file')
    parser.add_argument('-b', '--bleu', type=int, default='3', help='BLEU-N')

    args = parser.parse_args()

    single_file_evaluation(sources=args.sources, targets=args.targets, bleu_n=args.bleu)
