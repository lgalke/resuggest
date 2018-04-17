import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch import LongTensor as LT
from torch import FloatTensor as FT
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch import optim
import numpy as np

from .rnn import PaddedSequence

CUDA = torch.cuda.is_available()


class TripletAgent(object):
    def __init__(self, n_hidden=100, n_negative=5, n_epochs=5, p=2,
                 lr=3e-4, **tfidf_params):
        self.hidden_size = n_hidden
        self.to_ix = PaddedSequence(**tfidf_params)
        self.embedding = None
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=p)
        self.n_negative = n_negative
        self.n_epochs = n_epochs
        self.p = p
        self.lr = lr
        self.last_epoch = -1

    def __str__(self):
        return str(self.__dict__)

    def partial_fit(self, q_batch, a_batch):
        self.embedding1.train()
        self.embedding1.zero_grad()
        self.embedding2.train()
        self.embedding2.zero_grad()
        if CUDA:
            q_batch = q_batch.cuda()
            a_batch = a_batch.cuda()
        h_anchor = self.embedding1(q_batch)
        h_positive = self.embedding2(a_batch)
        # shift by one to get same amount of negatives
        head = h_positive[:1]
        tail = h_positive[1:]
        h_negative = torch.cat((tail, head), dim=0)
        loss = self.criterion(h_anchor, h_positive, h_negative)
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        print("\r[{}/{}] {:.4f}".format(self.last_epoch + 2, self.n_epochs,
                                        loss.data[0]), end='', flush=True)

    def fit(self, questions, answers):
        questions, answers = np.asarray(questions), np.asarray(answers)
        self.to_ix.fit(np.append(questions, answers))
        questions_numpy = self.to_ix.transform(questions)
        answers_numpy = self.to_ix.transform(answers)

        num_embeddings, hdim = self.to_ix.vocabulary_size_, self.hidden_size

        self.embedding1 = nn.Sequential(nn.EmbeddingBag(num_embeddings, hdim,
                                                        mode='mean'),
                                        nn.Linear(hdim, hdim),
                                        nn.Dropout(0.2), nn.ReLU(),
                                        nn.Linear(hdim, hdim),
                                        nn.Dropout(0.2), nn.ReLU(),
                                        nn.Linear(hdim, hdim))
        self.embedding2 = nn.Sequential(nn.EmbeddingBag(num_embeddings, hdim,
                                                       mode='mean'),
                                        nn.Linear(hdim, hdim),
                                        nn.Dropout(0.2), nn.ReLU(),
                                        nn.Linear(hdim, hdim),
                                        nn.Dropout(0.2), nn.ReLU(),
                                        nn.Linear(hdim, hdim))
        self.embedding1 = self.embedding1.cuda()
        self.embedding2 = self.embedding2.cuda()
        self.optimizer1 = optim.Adam(self.embedding1.parameters(), lr=self.lr)
        self.optimizer2 = optim.Adam(self.embedding2.parameters(), lr=self.lr)
        n_samples = questions_numpy.shape[0]
        batch_size = self.n_negative + 1
        try:
            self.last_epoch = -1
            for epoch in range(self.n_epochs):
                q_shuffled, a_shuffled = shuffle(questions_numpy,
                                                 answers_numpy)
                q_shuffled, a_shuffled = V(LT(q_shuffled)), V(LT(a_shuffled))
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    if end > n_samples:
                        break
                    q_batch = q_shuffled[start:end]
                    a_batch = a_shuffled[start:end]
                    self.partial_fit(q_batch, a_batch)
                self.last_epoch = epoch
        except KeyboardInterrupt:
            print("STAHP!.")
        finally:
            print()

        self.embedding1 = self.embedding1.cpu()
        self.embedding2 = self.embedding2.cpu()
        self.embedding2.eval()
        self._embedded_answers = self.embedding2(V(LT(answers_numpy)))
        self._answers = answers

    def predict(self, question, k=1):
        self.embedding1.eval()
        q_numpy = self.to_ix.transform([question])
        embedded_question = self.embedding1(V(LT(q_numpy)))
        dist = F.pairwise_distance(embedded_question.expand_as(self._embedded_answers), self._embedded_answers,
                                   p=self.p)
        values, indices = dist.topk(k, sorted=True, largest=False, dim=0)
        indices = indices.data.squeeze().numpy()
        values = values.data.squeeze().numpy()
        return self._answers[indices], values








# vim: set ft=python.torch:
