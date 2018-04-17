import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from align import negative_samples
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CUDA = torch.cuda.is_available()


class PaddedSequence(object):  # should subclass sklearn Estimator
    """ A Vectorizer that transforms text into padded sequences instead of bag
    of words / ngrams. Drop-in replacement for CountVectorizer /
    TfIdfVectorizer. CountVectorizer is used to build the vocabulary. """
    unk_ = 1
    pad_ = 0
    offset_ = 2

    def __init__(self, sort=False, fix_empty=True, drop_unk=False, **cv_params):
        """
        unk_token: If None, unknown tokens are dropped. Else the specified
        index will be used for unk tokens.

        Other keyword arguments are directly passed to CountVectorizer
        """
        super(PaddedSequence, self).__init__()
        assert 'ngram_range' not in cv_params, "This does not make sense."
        self.cv = CountVectorizer(**cv_params)
        self.fix_empty = fix_empty
        self.sort = sort
        self.drop_unk = drop_unk
        self.vocabulary_size_ = None

    def fit(self, raw_documents):
        """ Constructs vocabulary from raw documents """
        self.cv.fit(raw_documents)
        # store vocabulary size
        self.vocabulary_size_ = len(self.cv.vocabulary_) + self.offset_
        return self

    def transform(self, raw_documents, return_lengths=False):
        """ Transforms a batch of raw_documents and pads to max length within batch """

        vocab = self.cv.vocabulary_

        # Tokenize sentences
        analyze = self.cv.build_analyzer()
        sentences = (analyze(doc) for doc in raw_documents)  # generator

        if self.drop_unk:
            sentences = [[vocab[w] + self.offset_ for w in s if w in vocab] for s in sentences]
        else:
            sentences = [[vocab[w] + self.offset_ if w in vocab else self.unk_
                          for w in s] for s in sentences]

        if self.fix_empty:
            # Place a single unk token in empty sentences
            sentences = [s if s else [self.unk_] for s in sentences]

        if self.sort:
            sentences.sort(key=len, reverse=True)
            max_length = len(sentences[0])
        else:
            max_length = max(map(len, sentences))

        n_samples = len(sentences)
        padded_sequences = np.empty((n_samples, max_length), dtype='int64')
        padded_sequences.fill(self.pad_)
        for i, sentence in enumerate(sentences):
            for j, token in enumerate(sentence):
                padded_sequences[i, j] = token

        if return_lengths:
            lengths = list(map(len, sentences))
            return padded_sequences, lengths
        else:
            return padded_sequences

    def fit_transform(self, raw_documents, **transform_params):
        """ Applies fit, then transform on raw documents """
        return self.fit(raw_documents).transform(raw_documents, **transform_params)

    def inverse_transform(self, sequences, join=None):
        """ Inverse transforms an iterable of iterables holding indices """
        reverse_vocab = {idx + self.offset_: word for word, idx in self.cv.vocabulary_.items()}
        assert self.pad_ not in reverse_vocab
        assert self.unk_ not in reverse_vocab
        reverse_vocab[self.pad_] = '<PAD>'
        reverse_vocab[self.unk_] = '<UNK>'
        sentences = [[reverse_vocab[t] for t in s] for s in sequences]
        if join is None:
            return sentences
        else:
            join_str = str(join)
            return [join_str.join(s) for s in sentences]


class MatchingRNN(nn.Module):
    """
    Encodes a pair of sequences and computes similarity between hidden states
    """

    def __init__(self,
                 num_embeddings,  # vocabulary size
                 embedding_dim,  # embedding dim
                 hidden_size,  # amount of hidden units in RNN state
                 num_layers,  # number of hidden layers
                 similarity='cosine',
                 bidirectional=True,
                 padding_idx=None,
                 **rnn_kwargs):
        super(MatchingRNN, self).__init__()
        assert similarity in ['cosine', 'inner', 'bilinear', 'mlp']
        try:
            num_embeddings1, num_embeddings2 = num_embeddings
            shared_emb = False
        except TypeError:
            shared_emb = True

        if shared_emb:
            # also seperate paddings?
            self.emb = nn.Embedding(num_embeddings,
                                    embedding_dim,
                                    padding_idx=padding_idx)
        else:
            try:
                pad_idx1, pad_idx2 = padding_idx
            except TypeError:
                print("[warn] using same padding index for both embeddings")
                pad_idx1 = pad_idx2 = padding_idx
            print("padding indices", pad_idx1, pad_idx2)
            print("num embeddings1, 2 =", num_embeddings1, num_embeddings2)
            self.emb1 = nn.Embedding(num_embeddings1,
                                     embedding_dim,
                                     padding_idx=pad_idx1)
            self.emb2 = nn.Embedding(num_embeddings2,
                                     embedding_dim,
                                     padding_idx=pad_idx2)

        self.rnn1 = nn.GRU(embedding_dim if shared_emb else num_embeddings1,
                           hidden_size,
                           num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           **rnn_kwargs)
        self.rnn2 = nn.GRU(embedding_dim if shared_emb else num_embeddings2,
                           hidden_size,
                           num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           **rnn_kwargs)

        self.mlp = None
        self.bilin = None
        if similarity == 'bilinear':
            print("Biliniear similarity")
            h_dim = hidden_size * (2 if bidirectional else 1) * num_layers
            self.bilin = nn.Bilinear(h_dim, h_dim, 1)
        elif similarity == 'mlp':
            h_dim = hidden_size * (2 if bidirectional else 1) * num_layers
            self.mlp = nn.Sequential(nn.Linear(h_dim * 2, h_dim),
                                     nn.Dropout(0.2),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.Dropout(0.2),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.Dropout(0.2),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, 1, bias=False))
        self.bidirectional = bidirectional
        self.similarity = similarity

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.shared_embedding = shared_emb

    def cuda(self):
        self.rnn1 = self.rnn1.cuda()
        self.rnn2 = self.rnn2.cuda()
        if self.mlp is not None:
            self.mlp = self.mlp.cuda()
        if self.bilin is not None:
            self.bilin = self.bilin.cuda()
        return self

    def forward(self, *input):
        """ Encodes two inputs and computes a similarity scores """
        input1, input2 = input

        # apply embeddings and rnns
        h_1 = self.encode(1, input1)
        h_2 = self.encode(2, input2)

        # compute similarity and return
        return self.sim(h_1, h_2)

    def init_hidden(self, batch_size):
        num_directions = 2 if self.bidirectional else 1
        return V(torch.zeros(batch_size, self.num_layers * num_directions, self.hidden_size))

    def encode(self, which, input, use_cuda=CUDA):
        """ Encodes a batch of inputs with one of the two rnns """
        # Sort, store mask then unsort!
        inps, lens = input
        lens, perm_index = torch.sort(lens, descending=True)
        inps = inps[perm_index]
        if self.shared_embedding:
            embed = self.emb
        else:
            embed = {
                1: self.emb1,
                2: self.emb2
            }[which]
        inps = inps.cuda() if embed.weight.is_cuda else inps
        inps = embed(inps)

        # Prepare reverse index mapping to restore order
        __, reverse_index = torch.sort(perm_index)
        reverse_index = reverse_index.cuda() if use_cuda else reverse_index

        rnn = {
            1: self.rnn1,
            2: self.rnn2
        }[which]
        inps = inps.cuda() if use_cuda else inps
        packed_input = pack_padded_sequence(inps, list(lens), batch_first=True)
        # h0 = self.init_hidden(inps.size(0))
        # h0 = h0.cuda() if use_cuda else h0
        pack_out, hn = rnn(packed_input)
        # out: shape
        # hn: num_dirs * num_layers, bsz, n_hidden
        h = hn.view(hn.size(1), hn.size(0) * hn.size(2))
        # h, lens = pad_packed_sequence(pack_out, batch_first=True)
        # h = h.sum(dim=1)
        # finally "unsort"
        return h[reverse_index]


    def sim(self, h1, h2):
        """
        Aggregate h1 and h2 into joint score
        """
        if self.similarity == 'inner':
            sim = torch.bmm(h1.unsqueeze(1), h2.unsqueeze(2))
        elif self.similarity == 'cosine':
            sim = F.cosine_similarity(h1, h2)
        elif self.similarity == 'bilinear':
            sim = self.bilin(h1, h2)
        elif self.similarity == 'mlp':
            # this overfits dramatically
            sim = self.mlp(torch.cat((h1, h2), dim=1))
        else:
            return KeyError("Similarity unkown", self.similarity)
        return sim



class RNNAgent():
    """ Trains a matching RNN """
    def __init__(self,
                 cv_params={},
                 similarity='cosine',
                 lr=1e-3,
                 negative=1,
                 num_layers=1,
                 embedding_size=100,
                 hidden_size=100,
                 n_epochs=10,
                 share_vocab=True,
                 **rnn_params):
        if share_vocab:
            self.vect = PaddedSequence(sort=False, fix_empty=True, **cv_params)
        else:
            self.vect = (PaddedSequence(sort=False, fix_empty=True,
                                        **cv_params),
                         PaddedSequence(sort=False, fix_empty=True,
                                        **cv_params))
        self.share_vocab = share_vocab

        self.rnn_params = rnn_params
        print("Using similarity", similarity)
        self.similarity = similarity
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.negative = negative
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size

        self.criterion = torch.nn.SoftMarginLoss()
        # self.criterion = torch.nn.MSELoss()

        self.model, self.optim = None, None
        self._answers, self._encoded_answers = None, None

        self.last_epoch = None

    def __str__(self):
        fmtstr = "Matching-BiRNN-h{}-{}-n{}-lr{}".format(self.hidden_size,
                                                         self.similarity,
                                                         self.negative,
                                                         self.lr)
        fmtstr += ("-sharedVocab" if self.share_vocab else "-separateVocab")
        return fmtstr

    def partial_fit(self, q, a):
        """ Fits a batch of paired question-answer data """
        # encoding
        self.model.train()
        self.model.zero_grad()

        hq = self.model.encode(1, q)
        ha = self.model.encode(2, a)

        # positive
        sim = self.model.sim(hq, ha)
        tgt = V(FT([1])).expand_as(sim)
        tgt = tgt.cuda() if CUDA else tgt
        loss = self.criterion(sim, tgt)

        # negative
        tgt = V(FT([-1])).expand_as(sim)
        tgt = tgt.cuda() if CUDA else tgt
        for k in range(1, ha.size(0)):
            sim = self.model.sim(hq, torch.cat([ha[k:], ha[:k]], dim=0))
            loss += self.criterion(sim, tgt)

        loss /= ha.size(0)
        loss.backward()
        self.optim.step()
        print("\r[{}/{}] {:.4f}".format(self.last_epoch + 2, self.n_epochs,
                                        loss.data[0]), end='', flush=True)


    # def partial_fit_old(self, q_batch, a_batch):
    #     """ Fits a batch """
    #     self.model.train()
    #     self.model.zero_grad()

    #     enc_questions = self.model.encode(1, q_batch, use_cuda=CUDA)
    #     enc_answers = self.model.encode(1, a_batch, use_cuda=CUDA)

    #     # positive
    #     sim = self.model.sim(enc_questions, enc_answers)
    #     tgt = V(FT([1])).expand_as(sim)
    #     tgt = tgt.cuda() if CUDA else tgt
    #     loss = self.criterion(sim, tgt)


    #     batch_size = a_batch.size(0)

    #     neg_tgt = V(FT([-1])).expand_as(sim)
    #     neg_tgt = neg_tgt.cuda() if CUDA else neg_tgt

    #     for k in range(1, batch_size):
    #         neg_answers = torch.cat(enc_answers[k:], enc_answers[:k], dim=0)
    #         # a_neg = torch.cat([a_tensor[k:], a_tensor[:k]], dim=0)
    #         # a_neg_lens = a_lengths[k:] + a_lengths[:k]
    #         # sim = self.model(q, (a_neg, a_neg_lens))
    #         sim = self.model.sim(enc_questions, neg_answers)
    #         loss += self.criterion(sim, neg_tgt)

    #     loss /= batch_size


    #     print("\r[{}/{}] {:.4f}".format(self.last_epoch + 2, self.n_epochs,
    #                                     loss.data[0]), end='', flush=True)
    #     loss.backward()
    #     self.optim.step()


    def fit(self, questions, answers):
        """ Fits to list of paired question-answer data """
        questions, answers = np.asarray(questions), np.asarray(answers)
        if self.share_vocab:
            print("Using shared vocab")
            self.vect.fit(np.append(questions, answers))
            emb_dim = self.vect.vocabulary_size_
            pad_ids = self.vect.pad_

            q_all, q_all_lens = self.vect.transform(questions,
                                                    return_lengths=True)
            a_all, a_all_lens = self.vect.transform(answers,
                                                    return_lengths=True)
        else:
            print("Using separate vocabs")
            q_all, q_all_lens = self.vect[0].fit_transform(questions,
                                                           return_lengths=True)
            a_all, a_all_lens = self.vect[1].fit_transform(questions,
                                                           return_lengths=True)
            emb_dim = [vect.vocabulary_size_ for vect in self.vect]
            pad_ids = [vect.pad_ for vect in self.vect]

        self.model = MatchingRNN(emb_dim,
                                 self.embedding_size,
                                 self.hidden_size,
                                 self.num_layers,
                                 similarity=self.similarity,
                                 padding_idx=pad_ids,
                                 **self.rnn_params)
        if CUDA:
            self.model = self.model.cuda()
        # self.optim = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        print(*list(self.model.parameters()), sep='\n')
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optim = optim.ASGD(self.model.parameters(), lr=self.lr)

        batch_size = self.negative + 1
        n_samples = questions.shape[0]
        assert answers.shape[0] == n_samples

        try:
            self.last_epoch = -1
            for epoch in range(self.n_epochs):
                data = shuffle(q_all, q_all_lens, a_all, a_all_lens)
                assert len(data) == 4
                q_inps, q_lens = V(LT(data[0])), LT(data[1])
                a_inps, a_lens = V(LT(data[2])), LT(data[3])
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    q_batch = q_inps[start:end], q_lens[start:end]
                    a_batch = a_inps[start:end], a_lens[start:end]
                    self.partial_fit(q_batch, a_batch)
                self.last_epoch = epoch
        except KeyboardInterrupt:
            print("STAHP!.")
        finally:
            print()

        self.model = self.model.cpu()  # put stuff on CPU now
        self.model.eval()
        # maintain Order of the original input
        self._encoded_answers = self.model.encode(2, (V(LT(a_all)),
                                                      LT(a_all_lens)),
                                                  use_cuda=False)
        self._answers = answers

    def predict(self, question, k=3):
        """ Predicts """
        self.model.eval()
        q_inps, q_lens = self.vect.transform([question], return_lengths=True)
        q_inps, q_lens = V(LT(q_inps)), LT(q_lens)
        question_h = self.model.encode(1, (q_inps, q_lens), use_cuda=False)
        if self.similarity == 'bilinear' or self.similarity == 'mlp':
            question_h = question_h.expand_as(self._encoded_answers)
        sim = self.model.sim(question_h, self._encoded_answers)
        values, indices = sim.topk(k, dim=0, sorted=True, largest=True)
        indices = torch.squeeze(indices.data).numpy()
        values = torch.squeeze(values.data).numpy()
        return self._answers[indices], values



# vim: set ft=python.torch:
