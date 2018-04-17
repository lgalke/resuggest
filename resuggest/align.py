import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import LongTensor as LT
from torch import FloatTensor as FT

from .ub import GensimEmbeddedVectorizer

W2V_PATH = "/data21/lgalke/vectors/cc.de.300.vec"
W2V_IS_BINARY = False


CUDA = torch.cuda.is_available()
DTYPE = 'float32'


def negative_samples(a, b, negative_target=-1, dtype='int64'):
    """ Constructs negative samples for batch """
    bsz = a.shape[0]
    assert b.shape[0] == bsz, "Batch size does not match"
    if bsz == 1:
        print("Batch size 1 warning")
    # copy
    a_out = np.array(a)
    b_out = np.array(b)
    y_out = np.ones((bsz), dtype=dtype)

    for i in range(1, bsz):  # -1 to not roll over
        # Cycle through b values with negative target
        a_out = np.vstack([a_out, a])
        b_out = np.vstack([b_out, np.roll(b, i, axis=0)])

        y_tgt = np.empty((bsz), dtype=dtype)
        y_tgt.fill(negative_target)
        y_out = np.concatenate((y_out, y_tgt))

    return a_out, b_out, y_out


def to_numpy(inp, dtype=DTYPE):
    if sp.issparse(inp):
        inp = inp.toarray()
    return inp.astype(dtype)


def to_var(input, use_cuda=CUDA):
    """
    Transforms either numpy array or sparse matrix into
    V(torch.FloatTensor)
    """
    input = V(FT(input))
    if use_cuda:
        input = input.cuda()
    return input


def rowdot(a, b):
    """ Computes row-wise dot product of two batches
    >>> a = V(torch.FloatTensor(np.arange(6).reshape(2, 3)))
    >>> b = V(torch.FloatTensor(np.arange(6).reshape(2, 3)))
    >>> rowdot(a, b).data
    <BLANKLINE>
      5
     50
    [torch.FloatTensor of size 2]
    <BLANKLINE>
    """
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze()


class TanhTower(nn.Module):
    """
    A Tanh Tower of which multiple can be used to separately encode text
    """
    def __init__(self, n_inputs, n_hidden):
        super(TanhTower, self).__init__()
        self.lin1 = nn.Linear(n_inputs, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_hidden)
        self.lin4 = nn.Linear(n_hidden, n_hidden)
        self.drp1 = nn.Dropout(0.2)
        self.drp2 = nn.Dropout(0.2)
        self.drp3 = nn.Dropout(0.2)

    def forward(self, input):
        h = F.tanh(self.drp1(self.lin1(input)))
        h = F.tanh(self.drp2(self.lin2(h)))
        h = F.tanh(self.drp3(self.lin3(h)))
        return F.tanh(self.lin4(h))


class TanhDotProduct(nn.Module):
    """ Dotproduct model """
    def __init__(self, n_inputs, n_hidden, bilinear=False):
        super(TanhDotProduct, self).__init__()
        self.tower1 = TanhTower(n_inputs, n_hidden)
        self.tower2 = TanhTower(n_inputs, n_hidden)
        if bilinear:
            self.bilin = nn.Bilinear(n_hidden, n_hidden, 1)
        else:
            self.bilin = None

    def forward(self, x1, x2):
        hx1 = self.tower1(x1)
        hx2 = self.tower2(x2)
        return self.sim(hx1, hx2)

    def sim(self, h1, h2):
        if self.bilin is None:
            # Use cosine as default
            return F.cosine_similarity(h1, h2).view(-1, 1)
        return self.bilin(h1, h2)


class CatRelu(nn.Module):
    """ Joint model """
    def __init__(self, n_inputs, n_hidden):
        super(CatRelu, self).__init__()
        self.lin1 = nn.Linear(n_inputs * 2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_hidden)
        self.lin4 = nn.Linear(n_hidden, 1, bias=False)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)

    def forward(self, x1, x2):
        inp = torch.cat([x1, x2], dim=1)
        h = F.relu(self.drop1(self.lin1(inp)))
        h = F.relu(self.drop2(self.lin2(h)))
        h = F.relu(self.drop3(self.lin3(h)))
        # learned similarity should be compatible to cosine (in [-1,1])
        return self.lin4(h)

    def query(self, q, a):
        q = q.expand_as(a)
        return self.forward(q, a)


class NonlinearAlignmentAgent():
    def __init__(self,
                 negative=1,
                 n_epochs=5,
                 vectors=None,
                 n_hidden=100,
                 variant='joint',
                 lr=1e-3,
                 **tfidf_params):
        assert variant in ['joint', 'dotproduct', 'bilinear']
        if vectors:
            self.vectorizer = GensimEmbeddedVectorizer(vectors, **tfidf_params)
        else:
            self.vectorizer = TfidfVectorizer(**tfidf_params)

        self._vectors = vectors
        self.n_epochs = n_epochs
        self.negative = negative
        self.n_hidden = n_hidden
        self._fit_y = None
        # self.q_tower, self.a_tower = None, None
        # self.q_optim, self.a_optim = None, None
        self.model, self.optim = None, None
        # self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        # self.criterion = nn.CosineEmbeddingLoss(size_average=True, margin=0)
        # self.criterion = nn.HingeEmbeddingLoss(size_average=True, margin=1.0)
        self.criterion = nn.SoftMarginLoss()

        # self.similarity = {
        #     'cosine': torch.nn.CosineSimilarity(dim=1),
        #     'inner': rowdot
        # }[similarity]

        self._answers = None
        self._answer_vectors = None
        self._encoded_answers = None

        self._lr = lr
        self._variant = variant

    def __str__(self):
        desc = "NAAgent-{}-neg{}-hid{}-lr{}".format(self._variant,
                                                    self.negative,
                                                    self.n_hidden, self._lr)
        if self._vectors is not None:
            desc += "-vectors"
        return desc

    def get_params(self):
        """ Returns a string desription including most relevant hyperparams """
        return self.__dict__

    def _partial_fit_joint(self, x_q, x_a):
        self.model.train()
        self.model.zero_grad()
        x_q, x_a = to_numpy(x_q), to_numpy(x_a)  # to dense ndarrays
        # sample negatives
        x_q, x_a, tgt = negative_samples(x_q, x_a, dtype='float32')
        # make numpy stuff variable
        x_q, x_a, tgt = to_var(x_q), to_var(x_a), to_var(tgt)
        sim = self.model(x_q, x_a)

        loss = self.criterion(sim, tgt.view(-1, 1))
        print(loss.data[0])
        loss.backward()
        self.optim.step()

    def _partial_fit_dotproduct(self, x_q, x_a):
        """ In the dot product case, we can precompute all the stuff """
        self.model.train()
        self.model.zero_grad()
        # to dense, to variable
        x_q, x_a = to_var(to_numpy(x_q)), to_var(to_numpy(x_a))
        h_q = self.model.tower1(x_q)
        h_a = self.model.tower2(x_a)

        # Positive phase
        sim = self.model.sim(h_q, h_a)
        tgt = V(FT([1])).expand_as(sim)
        tgt = tgt.cuda() if CUDA else tgt
        loss = self.criterion(sim, tgt)

        # Negative phase
        tgt = V(FT([-1])).expand_as(sim)
        tgt = tgt.cuda() if CUDA else tgt
        for k in range(1, h_a.size(0)):
            # Compare negative samples within batch
            h_a_negative = torch.cat([h_a[k:], h_a[:k]], dim=0)
            sim = self.model.sim(h_q, h_a_negative)
            loss += self.criterion(sim, tgt)

        loss /= h_a.size(0)
        loss.backward()
        self.optim.step()
        print(loss.data[0])

    def partial_fit(self, x_q, x_a):
        """
        Fits a batch of vector representations of questions xq and answers xa
        """
        {
            'joint': self._partial_fit_joint,
            'dotproduct': self._partial_fit_dotproduct,
            'bilinear': self._partial_fit_dotproduct
        }[self._variant](x_q, x_a)

    def fit(self, questions, answers):
        """
        Fits to a bunch of questions and answers (aligned string iterables)
        """
        questions = np.asarray(questions)
        answers = np.asarray(answers)
        n_samples = questions.shape[0]
        assert answers.shape[0] == n_samples, "fit: X and y dont match (size)"
        self.vectorizer.fit(np.append(questions, answers))

        # Store vectorized QA pairs
        question_vectors = self.vectorizer.transform(questions)
        answer_vectors = self.vectorizer.transform(answers)

        # set n inputs
        if hasattr(self.vectorizer, 'embedding'):
            n_inputs = self.vectorizer.embedding.shape[1]
        else:
            n_inputs = len(self.vectorizer.vocabulary_)

        print("N_inputs", n_inputs)

        # Construct towers
        if self._variant == 'joint':
            self.model = CatRelu(n_inputs, self.n_hidden)
        elif self._variant == 'dotproduct':
            self.model = TanhDotProduct(n_inputs, self.n_hidden, bilinear=False)
        elif self._variant == 'bilinear':
            self.model = TanhDotProduct(n_inputs, self.n_hidden, bilinear=True)
        else:
            raise ValueError(self._variant + " not in 'joint', 'dotproduct'")

        if CUDA:
            print("CUDA is used")
            self.model = self.model.cuda()

        # and their optimizers
        self.optim = optim.Adam(self.model.parameters(), lr=self._lr)

        # batched calls to partial fit
        batch_size = self.negative + 1
        # + 1 so that number of negative samples is correct
        try:
            for epoch in range(self.n_epochs):
                print("Epoch", epoch+1)
                # shuffle
                shuffled_answer_vectors, shuffled_question_vectors = \
                    shuffle(answer_vectors, question_vectors)
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    if end > n_samples:
                        break
                    q_batch = shuffled_question_vectors[start:end]
                    a_batch = shuffled_answer_vectors[start:end]
                    self.partial_fit(q_batch, a_batch)
        except KeyboardInterrupt:
            print("Trained enough")
        finally:
            print()

        # Store answer vectors to reuse for prediction

        # self.q_tower = self.q_tower.cpu()
        # self.a_tower = self.a_tower.cpu()
        self.model = self.model.cpu()

        # Store answers and encoded answers for prediction
        if self._variant in {'dotproduct', 'bilinear'}:
            # self.a_tower.eval() # !
            self.model.eval()
            inp = to_var(to_numpy(answer_vectors), use_cuda=False)
            self._encoded_answers = self.model.tower2(inp)
            # inp = to_var(to_numpy(answer_vectors), use_cuda=False)
            # self._encoded_answers = self.a_tower(inp).data
        elif self._variant == 'joint':
            # store answer vectors for prediction
            self._answer_vectors = to_var(to_numpy(answer_vectors),
                                          use_cuda=False)

        self._answers = answers
        return self

    def predict(self, question, k=3):
        """
        Predict an answer for a given question by computing the similarity
        to all possible answers
        """
        self.model.eval()
        q_numpy = to_numpy(self.vectorizer.transform([question]))
        x_q = to_var(q_numpy, use_cuda=False)
        if self._variant == 'dotproduct':
            h_q = self.model.tower1(x_q)
            # encoded answers are precomputed
            sim = self.model.sim(h_q, self._encoded_answers)
        elif self._variant == 'bilinear':
            h_q = self.model.tower1(x_q)
            h_q = h_q.expand_as(self._encoded_answers)
            sim = self.model.sim(h_q, self._encoded_answers)
        elif self._variant == 'joint':
            sim = self.model.query(x_q, self._answer_vectors)

        # Compare to all possible answers and yield the ones with highest score
        values, indices = sim.topk(k, dim=0, sorted=True, largest=True)
        values = torch.squeeze(values.data).numpy()
        indices = torch.squeeze(indices.data).numpy()
        return self._answers[indices], values


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from gensim.models.keyedvectors import KeyedVectors
    from utils.conversation import input_loop
    from utils.datasets import load_parallel_text
    dframe = load_parallel_text(context_path='tmp/sources.de',
                                utterance_path='tmp/targets.de')

    print("Loading", W2V_PATH)
    # vectors = KeyedVectors.load_word2vec_format(W2V_PATH,
    # binary=W2V_IS_BINARY)
    # print(vectors)
    agent = NonlinearAlignmentAgent(variant='joint', negative=1, lr=3e-4,
                                    n_hidden=25)  # good

    agent.fit(dframe.Context.values, dframe.Utterance.values)
    input_loop(agent)


# vim: set ft=python.torch:
