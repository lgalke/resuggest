""" Response suggestion via Information Retrieval """
from vec4ir.core import Retrieval
from vec4ir.base import Matching, Tfidf
from vec4ir.word2vec import WordCentroidDistance
import numpy as np

USE_W2V = True
W2V_PATH = "/data21/lgalke/vectors/cc.de.300.vec"
# W2V_PATH = "/data21/lgalke/vectors/cc.de.300.bin.gz"
W2V_IS_BINARY = False

class RetrievalAgent(object):
    """
    Suggests response based on an information retrieval on the answers with the quesiton as query.
    Uses either Tfidf or WordCentroidDistance as Retrieval Model.
    """
    def __init__(self,
                 vectors=None,
                 use_matching=True,
                 use_idf=True,
                 lookup_context=False):
        self._use_matching = use_matching
        self._use_idf = use_idf
        self._lookup_context = lookup_context



        if use_matching:
            match_op = Matching() # Boolean OR matching
        else:
            match_op = None
        if vectors:
            retmodel = WordCentroidDistance(embedding=vectors, use_idf=use_idf)
            name = 'wcd'
        else:
            retmodel = Tfidf(use_idf=use_idf)
            name = 'tfidf'
        self.model = Retrieval(retmodel, name=name, matching=match_op)

    def __str__(self):
        desc = self.model.name
        if not self._use_matching:
            desc += '-nomatching'
        if not self._use_idf:
            desc += '-noidf'
        if self._lookup_context:
            desc += '-lookup_context'
        return desc

    def fit(self, raw_questions, raw_answers):
        """ Fits retrieval model on raw answers """
        if self._lookup_context:
            self.model.fit(raw_questions, np.asarray(raw_answers))
        else:
            self.model.fit(raw_answers, np.asarray(raw_answers))

    def predict(self, raw_question, k=3):
        """ Performs a query with the raw question """

        responses, scores = self.model.query(raw_question, k=k, return_scores=True)
        return responses, scores

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from gensim.models.keyedvectors import KeyedVectors
    from utils.conversation import input_loop
    from utils.datasets import load_parallel_text
    DFRAME = load_parallel_text(context_path='tmp/sources.de', utterance_path='tmp/targets.de')

    if USE_W2V:
        print("Loading", W2V_PATH)
        VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    else:
        VECTORS = None

    AGENT = RetrievalAgent(vectors=VECTORS, use_idf=True, use_matching=True)

    AGENT.fit(DFRAME.Context.values, DFRAME.Utterance.values)
    input_loop(AGENT)
