# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 04:35:35 2016

@author: lexx
"""

import glob
from time import strftime
#from datetime import datetime
from joblib import Parallel, delayed 
import datetime

import sys
import os
from spacy.en import English as _spEnglish, LOCAL_DATA_DIR as _spLOCAL_DATA_DIR
import numpy as np
import nltk

import theano
import theano.tensor as T

import lasagne
import lasagne.layers as LL

from nolearn.lasagne.visualize import plot_loss

from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne import TrainSplit
import nolearn

import lasagne.init as LI
import lasagne.nonlinearities as LN
import lasagne.objectives as LO
import lasagne.updates as LU

#sys.path.append("/Users/lexx/Documents/Work/python/")
#sys.path.append("/Users/lexx/Documents/Work/Kaggle/AllenAIScience/python/")
#sys.path.append("/Users/lexx/Documents/Work/python/Development/")
import AAI_data_3 as _aai_data
import help_scripts as _helpScripts

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from gensim.models import Word2Vec

SUBMIT = True

VERSION = 1
W2VEC_VERSION = 12
W2VEC_DIR = './data/w2v_models/{0}/'.format(W2VEC_VERSION)

XAPIAN_VERSION = 5
XAPIAN_DIR = './data/xapian/{0}/'.format(XAPIAN_VERSION)

VOCAB_SIZE = 10000
MAX_SENT_LENGTH = 40
MIN_SENT_LENGTH = 5
#THEMES_QUEST = 4
ARTICLES_QUEST = 5
MAX_SENT_ART = 50
#MAX_ART_THEME = 20
#RNN_TYPE = 'LSTM'

SENT_SIZE = 20
ART_END_TOK = 'ART_END_TOK'
UTOK = "UNKNOWN_TOKEN"    # unknown_token
PTOK = "PAD_TOKEN"

saveDir = './data/lasagneGRNN/{0}/'.format(VERSION)

def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)
    
def int16(k):
    return np.cast['int16'](k)


def flatten_tree_head(docs, tree_=None, i=0):
#    if i > 100:
#        print docs
#        return tree_
    i = i+1
    if tree_ is None:
        tree_ = []
    docs = list(docs)
    for ii, doc in enumerate(docs[:]):
        if doc not in tree_:
            if doc.head not in tree_ and doc.head != doc:
                docs.append(docs.pop(ii))
                flatten_tree(docs, tree_, i)
            else:
                tree_.append(doc)
    return tree_[::-1]


def flatten_sent(docs, root_ind, tree_=None):
    tree_ = []
    tree_.extend([docs[root_ind]])
    doc = list(docs[root_ind].subtree)
#    tree_.extend([docs.pop(root_ind)])
    i = 0
    while len(doc) > 1 and i < 10000:
        for tr in tree_:
            for tr_ch in tr.children:
                if tr_ch in doc:
                    tree_.extend([tr_ch])
                    doc.remove(tr_ch)
        i += 1
    return tree_[::-1]


def flatten_tree(docs, tree_=None):
    if tree_ is None:
        tree_ = []

    root_ind = get_root_ind(docs)
    for ri in root_ind:
        tree_.extend(flatten_sent(docs, ri))
    return tree_


def get_root_ind(doc):
    root_ind = []
    for i, doc_ in enumerate(doc):
        if doc_.dep_ == 'ROOT':
            root_ind.extend([i])
    return root_ind


def create_trees(doc, weights_):
    weights_tree = -1 * np.ones((len(doc), len(doc)), dtype='int32')
    root_ind = []
    words = [w.orth_.lower() for w in doc]
    for i, w in enumerate(doc):
        weights_tree[i][list(doc).index(w.head)] = weights_.index(w.dep_)
    weights_mask = np.ma.make_mask(weights_tree)
    root_ind = get_root_ind(doc)
#    for i, doc_ in enumerate(doc):
#        if doc_.dep_ == 'ROOT':
#            root_ind.extend([i])
    return words, weights_tree, root_ind


def create_trees_cumm(docs, weights_):
    words, weights_trees, root_inds = [], [], []
    for i, doc in enumerate(docs):
#        print doc
        doc_ = flatten_tree(doc)
        words_, weights_tree_, root_ind_ = create_trees(doc_, weights_)
        words.append(words_)
        weights_trees.append(weights_tree_)
        root_inds.append(root_ind_)
        _helpScripts.print_perc(float32(i)/float32(len(docs)) * 100 + 1)
    return words, weights_trees, root_inds


def combine_graph(words, weights_trees, pad_token='PAD_TOKEN', pad_value=-1):
    true_lengths = [(lambda x: len(x))(x) for x in words]
    pad_lengths = [max(true_lengths) - tl for tl in true_lengths]

    for i, _ in enumerate(words):
        weights_trees[i] = np.pad(weights_trees[i], ((0, pad_lengths[i]), (0, pad_lengths[i])), 'constant', constant_values=pad_value)
        words[i] += [pad_token for _ in range(pad_lengths[i])]
    return words, np.array(weights_trees)


def create_trees_all(text, nlp, vocab=set(), weights_=None, sent_group=None):
    """
    sent_group - list of index lists
    """
    if isinstance(text, basestring):
        sents = nltk.sent_tokenize(text)
        if sent_group:
            sents = [u' '.join([sents[si_]for si_ in si]) for si in sent_group]
    else:
        sents = text
    word_docs = [nlp(unicode(sent)) for sent in sents]
    word_docs = [[w for w in word_doc_ if w.orth_ not in [' ']] for word_doc_ in word_docs]

    words = [[w.orth_.lower() for w in ws] for ws in word_docs]
    poss = [[w.pos_ for w in ws] for ws in word_docs]

    heads = [[w.head for w in ws] for ws in word_docs]

    deps = [[w.dep_ for w in ws] for ws in word_docs]
    weights_ = list(set([d for w in deps for d in w]))
    vocab = list(set([d for w in words for d in w])) + [PTOK]
    words, weights_trees, root_inds = create_trees_cumm(word_docs, weights_)
    words, weights_trees = combine_graph(words, weights_trees, pad_token=PTOK)
    return words, weights_trees, vocab, root_inds, weights_


def get_root_ind_mask(root_inds, length):
    mask_ri = np.zeros((len(root_inds), length), dtype='int32')
    for i, j_ in enumerate(root_inds):
        for j in j_:
            mask_ri[i, j] = 1
    return mask_ri


def word_to_w2v(word, w2vec_model, PTOK='PAD_TOKEN'):
    try:
        return w2vec_model[word]
    except KeyError:
        repr_length = w2vec_model.syn0.shape[1]
        if word != PTOK:
            return np.ones((1, repr_length), dtype='float32')
        else:
            return np.zeros((1, repr_length), dtype='float32')


def sents_to_w2v(sent, w2vec_model, sent_l=None):
    repr_length = w2vec_model.syn0.shape[1]
    if not sent_l:
        sent_l = len(sent)
    result = np.zeros((sent_l, repr_length), dtype='float32')
    
    result_ = np.array([word_to_w2v(word, w2vec_model) for word in sent])
    for i, r in enumerate(result_):
        result[i] = r
    return result


def unflaffen_qa(data_qa, quest_n, answer_n):
#    train_qa = qa_data[:quest_n_train*(1 + answer_n)]
    data_q = data_qa[:quest_n]
    data_a = np.squeeze(data_qa[quest_n:].reshape((data_q.shape[0], 4, data_qa.shape[1], -1)))
    return data_q, data_a


def preprocess_qa(source, nlp, w2vec_model, row_num=None, test=False):
    answer_n = len(source.answer_names)
    if row_num is None:
        quest_n_train = source.train_data.shape[0]
        quest_n_test = source.test_data.shape[0]
    else:
        quest_n_train, quest_n_test = row_num, row_num
    _helpScripts.print_msg('generate tree representations for qa')
    train_questions = source.train_data['question'].values[:row_num]
    train_answers = source.train_data[source.answer_names].values[:row_num]
    if test:
        test_questions = source.test_data['question'].values[:row_num]
        test_answers = source.test_data[source.answer_names].values[:row_num]

        sents_all = np.concatenate((train_questions, train_answers.flatten(),
                                    test_questions, test_answers.flatten()))
    else:
        sents_all = np.concatenate((train_questions, train_answers.flatten()))
    words, weights_trees, vocab, root_inds, weights_ = create_trees_all(sents_all, nlp)
    word_n = np.shape(weights_trees)[1]
    wc_num = len(weights_)

    mask_root_ind = get_root_ind_mask(root_inds, length=word_n)
    mask = np.zeros_like(mask_root_ind)
    for i, ri in enumerate(root_inds):
        mask[i, range(ri[-1] + 1)] = 1
    input_ = np.array([sents_to_w2v(sent, w2vec_model, word_n) for sent in words], dtype='float32')
    train_q, train_a = unflaffen_qa(input_[:quest_n_train*(1 + answer_n)],
                                           quest_n_train, answer_n)

    mask_train_ri_q, mask_train_ri_a = unflaffen_qa(mask_root_ind[:quest_n_train*(1 + answer_n)],
                                                                  quest_n_train, answer_n)
    mask_train_q, mask_train_a = unflaffen_qa(mask[:quest_n_train*(1 + answer_n)],
                                                   quest_n_train, answer_n)
    wt_train_q, wt_train_a = unflaffen_qa(weights_trees[:quest_n_train*(1 + answer_n)],
                                                        quest_n_train, answer_n)
    if test is None:
        test_q, test_a = None, None
        mask_test_ri_q, mask_test_ri_a = None, None
        mask_test_q, mask_test_a = None, None
        wt_test_q, wt_test_a = None, None
    else:
        test_q, test_a = unflaffen_qa(input_[quest_n_train*(1 + answer_n):],
                                             quest_n_train, answer_n)
        mask_test_ri_q, mask_test_ri_a = unflaffen_qa(mask_root_ind[quest_n_train*(1 + answer_n):],
                                                                      quest_n_train, answer_n)
        mask_test_q, mask_test_a = unflaffen_qa(mask[quest_n_train*(1 + answer_n):],
                                                       quest_n_train, answer_n)
        wt_test_q, wt_test_a = unflaffen_qa(weights_trees[quest_n_train*(1 + answer_n):],
                                                            quest_n_train, answer_n)

    correct_answ_ = source.train_data['correctAnswer'].values[:row_num]
    targets = int32([np.where(answ == np.array(source.answer_sym))[0][0] for answ in correct_answ_])
    return (train_q, train_a, test_q, test_a,
            wt_train_q, wt_train_a, wt_test_q, wt_test_a,
            mask_train_q, mask_train_a, mask_test_q, mask_test_a,
            mask_train_ri_q, mask_train_ri_a, mask_test_ri_q, mask_test_ri_a,
            vocab, word_n, wc_num, targets)


class GRNNLayer(LL.MergeLayer):
    """

    incomings:  sentence embeddings (batch_s, word_n, emb_size)
                seq_mask (batch_s, word_n)
#                leave_ind (batch_s, n)
                weights_tree (batch_s, word_n, word_n)
                weights_speech_member (batch_s, word_n)
                ACT (batch_s, word_n, emb_size)
                ACT_ (batch_s, word_n, word_n, emb_size)
    """
    def __init__(self, incomings, emb_size, hidden_size, word_n, wc_num, wsm_num=1,
                 only_return_final=False,
                 WC=lasagne.init.Normal(std=0.1),
                 WSM=lasagne.init.Normal(std=0.1),
                 b=lasagne.init.Constant(1.0), dropout=0, **kwargs):
        super(GRNNLayer, self).__init__(incomings, **kwargs)
#        self.batch_size = batch_size
        self.wc_num = wc_num
        self.wsm_num = wsm_num
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.word_n = word_n
        self.only_return_final = only_return_final
        self.WC = self.add_param(WC, (self.wc_num, self.hidden_size, self.hidden_size), name="WC", regularizable=True)
#        if self.wsm_num > 1:
#            self.WSM = self.add_param(WSM, (self.wp_num, self.emb_size, self.emb_size), name="WP", regularizable=True)
#        else:
        self.WSM = self.add_param(WSM, (self.emb_size, self.hidden_size), name="WSM", regularizable=True)
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.hidden_size,), name="b",
                                    regularizable=False)
#        self.dropout = dropout
#        self.retain_prob = 1.0 - self.dropout
#        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        if self.only_return_final:
            return (input_shapes[0][0], self.hidden_size)
        else:
            return (input_shapes[0][0], input_shapes[0][1], self.hidden_size)

    def get_output_for(self, inputs, **kwargs):
        in_se = inputs[0]
        in_mask = T.cast(inputs[1], 'int32')
        in_mask = in_mask.dimshuffle((1, 0, 'x'))
        in_mask = T.repeat(in_mask, self.hidden_size, axis=2)
        in_weights_tree = T.cast(inputs[2], 'int32')
        ACT = inputs[3]
        ACT_ = inputs[4]
        if len(inputs) > 5:
            in_weights_sm = inputs[5]
        seg_max = T.max(T.sum(in_mask, axis=0))
        seq = T.arange(T.max(seg_max))     # sequence to scan

        def step(i, in_mask, ACT, ACT_, in_se, WT):
            sub_tree_idx_ = T.nonzero(WT[:, i, :] > -1)
            a_ = T.dot(in_se[:, i], self.WSM)  # + self.b
            if self.b is not None:
                a_ += self.b.dimshuffle('x', 0)
            a_ = a_ + T.sum(ACT_[:, i], axis=1)
            a_ = T.tanh(a_)
#            if self.dropout:
#                a_ = a_ / self.retain_prob * self._srng.binomial(a_.shape, p=self.retain_prob,
#                                                                 dtype=theano.config.floatX)
            a_ = T.switch(in_mask, a_, ACT[:, i-1])
            a__ = T.batched_tensordot(a_[sub_tree_idx_[0], :],
                                      self.WC[WT[sub_tree_idx_[0], 
                                                 i, sub_tree_idx_[1]]], axes=1)
#            if self.dropout:
#                a__ = a__ / self.retain_prob * self._srng.binomial(a__.shape, p=self.retain_prob,
#                                                                   dtype=theano.config.floatX)
            newACT_ = T.set_subtensor(ACT_[sub_tree_idx_[0], sub_tree_idx_[1], i],
                                      a__)
            newACT = T.set_subtensor(ACT[:, i], a_)
            return newACT, newACT_


        activations, _ = theano.scan(fn=step,
                                     sequences=[seq, in_mask],
                                     outputs_info=[ACT, ACT_],
                                     non_sequences=[in_se, in_weights_tree])
        if self.only_return_final:
            activations = activations[0][-1, :, seg_max-1, :]
        else:
            activations = activations[0][-1]
        return activations


def make_grnn(batch_size, emb_size, g_hidden_size, word_n,
              wc_num, dence, wsm_num=1, rnn_type='LSTM',
              rnn_size=12, dropout_d=0.5,# pooling='mean',
              quest_na=4, gradient_steps = -1,
              valid_indices=None, lr=0.05, grad_clip=10):

    def select_rnn(x):
        return {
            'RNN': LL.RecurrentLayer,
            'LSTM': LL.LSTMLayer,
            'GRU': LL.GRULayer,
        }.get(x, LL.LSTMLayer)
    
#    dence = dence + [1]
    RNN = select_rnn(rnn_type)
#------------------------------------------------------------------input layers
    layers = [
        (LL.InputLayer, {'name': 'l_in_se_q', 'shape': (None, word_n, emb_size)}),
        (LL.InputLayer, {'name': 'l_in_se_a', 'shape': (None, quest_na, word_n, emb_size)}),
        (LL.InputLayer, {'name': 'l_in_mask_q', 'shape': (None, word_n)}),
        (LL.InputLayer, {'name': 'l_in_mask_a', 'shape': (None, quest_na, word_n)}),
        (LL.InputLayer, {'name': 'l_in_mask_ri_q', 'shape': (None, word_n)}),
        (LL.InputLayer, {'name': 'l_in_mask_ri_a', 'shape': (None, quest_na, word_n)}),
        (LL.InputLayer, {'name': 'l_in_wt_q', 'shape': (None, word_n, word_n)}),
        (LL.InputLayer, {'name': 'l_in_wt_a', 'shape': (None, word_n, quest_na, word_n)}),
        (LL.InputLayer, {'name': 'l_in_act_', 'shape': (None, word_n, g_hidden_size)}),
        (LL.InputLayer, {'name': 'l_in_act__', 'shape': (None, word_n, word_n, g_hidden_size)}),
    ]
#------------------------------------------------------------------slice layers
#    l_qs = []
#    l_cas = []
    l_ase_names = ['l_ase_{}'.format(i) for i in range(quest_na)]
    l_amask_names = ['l_amask_{}'.format(i) for i in range(quest_na)]
    l_amask_ri_names = ['l_amask_ri_{}'.format(i) for i in range(quest_na)]
    l_awt_names = ['l_awt_{}'.format(i) for i in range(quest_na)]
    for i in range(quest_na):
        layers.extend([(LL.SliceLayer, {'name': l_ase_names[i], 'incoming': 'l_in_se_a',
                                        'indices': i, 'axis': 1})])
    for i in range(quest_na):
        layers.extend([(LL.SliceLayer, {'name': l_amask_names[i], 'incoming': 'l_in_mask_a',
                                        'indices': i, 'axis': 1})])
    for i in range(quest_na):
        layers.extend([(LL.SliceLayer, {'name': l_amask_ri_names[i], 'incoming': 'l_in_mask_ri_a',
                                        'indices': i, 'axis': 1})])
    for i in range(quest_na):
        layers.extend([(LL.SliceLayer, {'name': l_awt_names[i], 'incoming': 'l_in_wt_a',
                                        'indices': i, 'axis': 1})])
#-------------------------------------------------------------------GRNN layers
    WC = theano.shared(np.random.randn(wc_num, g_hidden_size, g_hidden_size).astype('float32'))
#    WC = LI.Normal(0.1)
    WSM = theano.shared(np.random.randn(emb_size, g_hidden_size).astype('float32'))
    b = theano.shared(np.ones(g_hidden_size).astype('float32'))
#    b = lasagne.init.Constant(1.0)
    layers.extend([(GRNNLayer, {'name': 'l_q_grnn',
                                'incomings': ['l_in_se_q', 'l_in_mask_q', 'l_in_wt_q', 'l_in_act_', 'l_in_act__'],
                                'emb_size': emb_size, 'hidden_size': g_hidden_size,
                                'word_n': word_n, 'wc_num': wc_num, 'wsm_num': wsm_num,
                                'only_return_final': False,
                                'WC': WC, 'WSM': WSM, 'b': b})])
    l_a_grnns_names = ['l_a_grnn_{}'.format(i) for i in range(quest_na)]
    for i, l_a_grnns_name in enumerate(l_a_grnns_names):
        layers.extend([(GRNNLayer, {'name': l_a_grnns_name,
                                    'incomings': [l_ase_names[i], l_amask_names[i], l_awt_names[i], 'l_in_act_', 'l_in_act__'],
                                    'emb_size': emb_size, 'hidden_size': g_hidden_size,
                                    'word_n': word_n, 'wc_num': wc_num, 'wsm_num': wsm_num,
                                    'only_return_final': False,
                                    'WC': WC, 'WSM': WSM, 'b': b})])
#------------------------------------------------------------concatenate layers
    layers.extend([(LL.ConcatLayer, {'name': 'l_qa_concat',
                                     'incomings': ['l_q_grnn'] + l_a_grnns_names})])
    layers.extend([(LL.ConcatLayer, {'name': 'l_qamask_concat',
                                     'incomings': ['l_in_mask_ri_q'] + l_amask_ri_names})])
#--------------------------------------------------------------------RNN layers
    layers.extend([(RNN, {'name': 'l_qa_rnn_f', 'incoming': 'l_qa_concat',
                          'mask_input': 'l_qamask_concat',
                          'num_units': rnn_size,
                          'backwards': False, 'only_return_final': True,
                          'grad_clipping': grad_clip})])
    layers.extend([(RNN, {'name': 'l_qa_rnn_b', 'incoming': 'l_qa_concat',
                          'mask_input': 'l_qamask_concat',
                          'num_units': rnn_size,
                          'backwards': True, 'only_return_final': True,
                          'grad_clipping': grad_clip})])
    layers.extend([(LL.ElemwiseSumLayer, {'name': 'l_qa_rnn_conc',
                                          'incomings': ['l_qa_rnn_f', 'l_qa_rnn_b']})])
##-----------------------------------------------------------------pooling layer
##    l_qa_pool = layers.extend([(LL.ExpressionLayer, {'name': 'l_qa_pool',
##                                                     'incoming': l_qa_rnn_conc,
##                                                     'function': lambda X: X.mean(-1),
##                                                     'output_shape'='auto'})])
#------------------------------------------------------------------dence layers
    l_dence_names = ['l_dence_{}'.format(i) for i, _ in enumerate(dence)]
    if dropout_d:
        layers.extend([(LL.DropoutLayer, {'name': 'l_dence_do' + 'do', 'p': dropout_d})])
    for i, d in enumerate(dence):
        if i < len(dence) - 1:
            nonlin = LN.tanh
        else:
            nonlin = LN.softmax
        layers.extend([(LL.DenseLayer, {'name': l_dence_names[i], 'num_units': d,
                                        'nonlinearity': nonlin})])
        if i < len(dence) - 1 and dropout_d:
            layers.extend([(LL.DropoutLayer, {'name': l_dence_names[i] + 'do', 'p': dropout_d})])


    def loss(x, t):
        return LO.aggregate(LO.categorical_crossentropy(T.clip(x, 1e-6, 1. - 1e-6), t))
#        return LO.aggregate(LO.squared_error(T.clip(x, 1e-6, 1. - 1e-6), t))

    if isinstance(valid_indices, np.ndarray) or isinstance(valid_indices, list):
        train_split=TrainSplit_indices(valid_indices=valid_indices)
    else:
        train_split=TrainSplit(eval_size=valid_indices, stratify=False)
    nnet = NeuralNet(
            y_tensor_type=T.ivector,
            layers=layers,
            update=LU.adagrad,
            update_learning_rate=lr,
#            update_epsilon=1e-7,
            objective_loss_function=loss,
            regression=False,
            verbose=2,
            batch_iterator_train=PermIterator(batch_size=batch_size),
            batch_iterator_test=BatchIterator(batch_size=batch_size/2),
#            batch_iterator_train=BatchIterator(batch_size=batch_size),
#            batch_iterator_test=BatchIterator(batch_size=batch_size),            
            #train_split=TrainSplit(eval_size=eval_size)
            train_split=train_split
        )
    nnet.initialize()
    PrintLayerInfo()(nnet)
    return nnet


class PermIterator(BatchIterator):
    def __init__(self, batch_size):
        super(PermIterator, self).__init__(batch_size)

    def transform(self, Xb, yb):
        q_in = Xb['l_q_in']
        q_in_mask = Xb['l_q_in_mask']
        c_in = Xb['l_c_in']
        c_in_mask = Xb['l_c_in_mask']
        qa_new_idx = np.random.permutation(q_in.shape[1])
        cont_new_idx = np.random.permutation(c_in.shape[1])
        X_new = {'l_q_in': q_in[:, qa_new_idx], 'l_q_in_mask': q_in_mask[:, qa_new_idx],
                 'l_c_in': c_in[:, cont_new_idx], 'l_c_in_mask': c_in_mask[:, cont_new_idx]}
        y_new = int32([np.where(yb[i] == qa_new_idx)[0][0] for i in range(yb.shape[0])])
        return X_new, y_new


class TrainSplit_indices(TrainSplit):
    def __init__(self, valid_indices):
        TrainSplit.__init__(self, eval_size=0)
        self.valid_indices = valid_indices

    def __call__(self, X, y, net):
        if np.any(self.valid_indices):
            all_indices = np.arange(y.shape[0])
            train_indices = np.array(list(set(all_indices) - set(self.valid_indices)))
#            return train_indices
            X_train, y_train = nolearn.lasagne.base._sldict(X, train_indices), y[train_indices]
            X_valid, y_valid = nolearn.lasagne.base._sldict(X, self.valid_indices), y[self.valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = nolearn.lasagne.base._sldict(X, slice(len(y), None)), y[len(y):]
        return X_train, X_valid, y_train, y_valid


def encode_input(encode_layer, X):
    """
    from https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.ipynb
    """
    return LL.get_output(encode_layer, inputs=X).eval()


def get_layer_by_name(net, name):
    """
    from https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.ipynb
    """
    for i, layer in enumerate(net.get_all_layers()):
        if layer.name == name:
            return layer, i
    return None, None


if __name__ == '__main__':
    _helpScripts.print_msg('start at time ' + strftime("%Y-%m-%d %H:%M:%S"))
#    print strftime("%Y-%m-%d %H:%M:%S")
    startTime = datetime.datetime.now()

#    os.chdir('/Users/lexx/Documents/Work/Kaggle/AllenAIScience/')
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    nlp = _spEnglish(data_dir=os.environ.get('SPACY_DATA', _spLOCAL_DATA_DIR))
#--------------------------------------------------------------load w2vec model
    w2vec_model_name = W2VEC_DIR + 'm{}'.format(W2VEC_VERSION)
    _helpScripts.print_msg('load the word2vec model from', w2vec_model_name)
    w2vec_model = Word2Vec.load(w2vec_model_name)
    w2vec_model.init_sims(replace=True)
    repr_length = w2vec_model.syn0.shape[1]
#--------------------------------------------------generate / load context data
#-------------------------------------------------------------------make source
    _helpScripts.print_msg('make source')
    source = _aai_data.Source(lower=False, create_sents=False)
    quest_n_train = source.train_data.shape[0]
    quest_n_test = source.test_data.shape[0]
    print 'there are {} train questions and {} test questions'.format(quest_n_train, quest_n_test)
#-----------------------------------------------------------------preprocess qa
    (train_q, train_a, test_q, test_a,
     wt_train_q, wt_train_a, wt_test_q, wt_test_a,
     mask_train_q, mask_train_a, mask_test_q, mask_test_a,
     mask_train_ri_q, mask_train_ri_a, mask_test_ri_q, mask_test_ri_a,
     vocab, word_n, wc_num, train_targets) = preprocess_qa(source, nlp, w2vec_model, row_num=2500, test=True)
#-----------------------------------------------------------------generate GRNN
    emb_size = repr_length
    rnn_type = 'LSTM'
    rnn_size = 50
#    dropout = 0.3
    train_indices = np.array(source.trainInd)
    valid_indices = np.array(source.testInd)
    BATCH_SIZE = 8
    N_EPOCHS = 100
    g_hidden_size = 100
#    NUM_HOPS = 3
#    ART_SEL = 4
#    ARTICLES_QUEST_ = 5
    lr = 0.01
    dence = [100, 50, 4]

    activations_ = np.zeros((train_q.shape[0], wt_train_q.shape[1], g_hidden_size), dtype='float32')
    activations__ = np.zeros((train_q.shape[0], wt_train_q.shape[1], wt_train_q.shape[1], g_hidden_size), dtype='float32')

    _helpScripts.print_msg('create RNN')
    print 'batch_size: {}, n_epochs: {}'.format(BATCH_SIZE, N_EPOCHS)
    print 'emb_size: {}'.format(emb_size)
    net = make_grnn(BATCH_SIZE, emb_size, g_hidden_size, word_n,
                    wc_num, dence, wsm_num=1, rnn_type='LSTM',
                    rnn_size=40, dropout_d=0.9,# pooling='mean',
                    quest_na=4, gradient_steps = -1,
                    valid_indices=0.2, lr=0.05, grad_clip=10)
#----------------------------------------------------------------------train NN
    _helpScripts.print_msg('train NN')
    net.fit({'l_in_se_q': train_q, 'l_in_se_a': train_a,
             'l_in_mask_q': float32(mask_train_q), 'l_in_mask_a': float32(mask_train_a),
             'l_in_mask_ri_q': float32(mask_train_ri_q), 'l_in_mask_ri_a': float32(mask_train_ri_a),
             'l_in_wt_q': float32(wt_train_q), 'l_in_wt_a': float32(wt_train_a),
             'l_in_act_': activations_, 'l_in_act__': activations__},
            train_targets, epochs=N_EPOCHS)

    plot_loss(net)


