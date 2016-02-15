
import glob
import time
from time import strftime
#from datetime import datetime
from joblib import Parallel, delayed 
import datetime

import multiprocessing

import os
os.environ['OMP_NUM_THREADS'] = '8'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import cPickle
import sys
import itertools
import pandas as pd

import numpy as np
import scipy
import pandas
#from numpy import fft
# Lasagne (& friends) imports
import theano
import theano.tensor as T

import lasagne
import lasagne.layers as LL
import lasagne.init as LI
import lasagne.nonlinearities as LN
import lasagne.objectives as LO
import lasagne.updates as LU

from lasagne.layers import get_output
from sklearn.preprocessing import LabelBinarizer
#from lasagne.objectives import aggregate

from sklearn.cross_validation import ShuffleSplit

sys.path.append("/Users/lexx/Documents/Work/python/")
sys.path.append("/Users/lexx/Documents/Work/Kaggle/AllenAIScience/python/")
#sys.path.append("/Users/lexx/Documents/Work/python/Development/")
import AAI_data_3 as _aai_data
import help_scripts as _helpScripts
#import lasagne_new_layers as _lnl

from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne import TrainSplit
import nolearn
from nolearn.lasagne.visualize import plot_loss

import nltk
import logging

SUBMIT = True

VERSION = 7
W2VEC_VERSION = 12
W2VEC_DIR = './data/w2v_models/{0}/'.format(W2VEC_VERSION)

XAPIAN_VERSION = 5
XAPIAN_DIR = './data/xapian/{0}/'.format(XAPIAN_VERSION)

VOCAB_SIZE = 10000
MAX_SENT_LENGTH = 40
MIN_SENT_LENGTH = 5
#THEMES_QUEST = 4
ARTICLES_QUEST = 10
MAX_SENT_ART = 10
#MAX_ART_THEME = 20
#RNN_TYPE = 'LSTM'

SENT_SIZE = 20
ART_END_TOK = 'ART_END_TOK'
UTOK = "UNKNOWN_TOKEN"    # unknown_token
PTOK = "PAD_TOKEN"

saveDir = './data/lasagneMNN/{0}/'.format(VERSION)

#wiki_dir = u'./data/corpus/wiki_text/'
#ck12_full_theme_dir = u'./data/corpus/ck12_full_themes/'
#ck12_full_dir = u'./data/corpus/ck12_full_spl_50/'

CORE_NUM = multiprocessing.cpu_count()


def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)


def int16(k):
    return np.cast['int16'](k)


def train_test_split(train_l, train_fr=0.8, seed=156):
    sss = ShuffleSplit(n=train_l, n_iter=1,
                       test_size=1.-train_fr,
                       random_state=seed)
    trainInd, testInd = zip(*sss)
    trainInd, testInd = list(trainInd[0]), list(testInd[0])
    return trainInd, testInd


def one_hot_enc(sent_inds, sent_size=SENT_SIZE, vocab_size=VOCAB_SIZE):
    print 'one hot encoding of the inputs'
    X = np.zeros((np.shape(sent_inds)[0], sent_size, vocab_size),
                 dtype='float32')
#    print np.shape(X)
    for j in range(np.shape(sent_inds)[0]):
        for i in range(np.shape(sent_inds)[1]):
            X[j,i,sent_inds[j][i]] = 1.0
    return X


def tokenize_doc(fileName,
                 max_sent_length=MAX_SENT_LENGTH,
                 min_sent_length=MIN_SENT_LENGTH):
    with open (fileName, "r") as myfile:
        text = myfile.read()
    text = text.decode('utf-8').lower()
#    for i in reversed(a):
    sentences = nltk.sent_tokenize(text)
    for sent in reversed(sentences[:]):
        if sent.startswith(u'click on') or\
           u'http' in sent:
            sentences.remove(sent)
    
    result = [nltk.word_tokenize(sent) for sent in sentences]
    if max_sent_length:
        for sent in reversed(result[:]):
            if len(sent) > max_sent_length:
                result.remove(sent)
    if min_sent_length:
        for sent in reversed(result[:]):
            if len(sent) < min_sent_length:
                result.remove(sent)
    return result


def tokenize_docs(dirName, mode='extend', add_sep=True,
                  max_sent_length=MAX_SENT_LENGTH,
                  min_sent_length=MIN_SENT_LENGTH):
    print 'max sentence length is {}, min {}'.format(max_sent_length, min_sent_length)
    if isinstance(dirName, basestring):
        files = glob.glob(dirName + '*.txt')
    else:
        files = dirName
    token_sent_ = []
    token_sent_ += Parallel(n_jobs=CORE_NUM, verbose=2)\
                           (delayed(tokenize_doc)(file_name, max_sent_length=max_sent_length,
                                                 min_sent_length=min_sent_length)
                           for file_name in files)
    if mode == 'extend':
        token_sent = []
        for ts_ in token_sent_:
            token_sent.extend(ts_)
            if add_sep:
                token_sent.extend([ART_END_TOK])
    else:
        return token_sent_, files


def list2d2np(list2d):
    result = np.array([], dtype='float32')
    for j, w in enumerate(list2d):
        if j == 0:
            result = w.reshape(1, -1)
        else:
            result = np.append(result,
                               w.reshape(1, -1), axis=0)
    return result


def word2ind(word, vocab, utok_ind=None):
    ind = np.where(vocab == unicode(word))
    if len(ind[0]) == 0:
        if not utok_ind:
            utok_ind = np.where(vocab == UTOK)
        ind = utok_ind
    return ind[0][0]


def words2ind(words, vocab, utok_ind=None):
    if not utok_ind:
        utok_ind = np.where(vocab == UTOK)
    return [word2ind(word, vocab, utok_ind) for word in words]


def sents2ind(sentences, vocab, max_l=None):
    utok_ind = np.where(vocab == UTOK)
    ptok_ind = np.where(vocab == PTOK)[0][0]
    result_ = [words2ind(words, vocab, utok_ind) for words in sentences]
    
    if not max_l:
        length = len(sorted(result_, key=len, reverse=True)[0])
    else:
        length = max_l
#    print 'max length of sentence', length
    result = np.array([xi+[ptok_ind]*(length-len(xi)) for xi in result_],
                       dtype='int32')
    return result


def sents2indPar(sentences, vocab, proc=8, max_l=None):
    utok_ind = np.where(vocab == UTOK)
    ptok_ind = np.where(vocab == PTOK)[0][0]
    result_ = Parallel(n_jobs=proc, verbose=2)(delayed(words2ind)(words,
                                                               vocab, utok_ind)
                                    for words in sentences)

    if not max_l:
        length = len(sorted(result_, key=len, reverse=True)[0])
    else:
        length = max_l
    print 'max length of sentence', length
    result = np.array([xi+[ptok_ind]*(length-len(xi)) for xi in result_],
                       dtype='int32')
    return result


def select_themes(wh_scores, wh_themes, themes_quest=ARTICLES_QUEST):
    """
    returns the themes according to the ranking
    firstly wil be the best themes for each question returned and then according to the summ scores of the theme
    """
    quest_n = len(wh_themes)
    themes = np.zeros((quest_n, themes_quest), dtype=object)
    for q in range(quest_n):
        q_themes_best = set(wh_themes[q][:, 0])
        themes[q, :len(q_themes_best)] = np.array(list(q_themes_best))
        q_themes_rest = np.array(list(set(wh_themes[q].flatten()) - q_themes_best), dtype=object)
        th_scores = np.array([np.sum(wh_scores[q][np.where(wh_themes[q] == q_themes_rest[i])]) for i in range(len(q_themes_rest))])
        th_scores_sorted = np.argsort(th_scores)[::-1]
        themes[q, len(q_themes_best):themes_quest] = q_themes_rest[th_scores_sorted[0:themes_quest-len(q_themes_best)]]
        _helpScripts.print_perc(float32(q)/float32(quest_n) * 100 + 1)
    return themes


def chunks(l, n):
    """Yield successive n-sized chunks from l.
    from http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_art_word_ind(sentences, vocab, sent_art, sent_l, pad_token_ind):
    """
    generates word indexies for articles
    """
    art_count = len(sentences)
    article_inds = np.ones((art_count,
                            sent_art, sent_l), dtype='int16') * pad_token_ind
                                 
    for art_ind in range(art_count):
        article = sentences[art_ind]
        try:
            sent_ind = sents2ind(article, vocab)
        except IndexError:
            continue
        article_inds[art_ind, :sent_ind.shape[0], :sent_ind.shape[1]] = sent_ind[:, :]                            

        _helpScripts.print_perc(float32(art_ind)/float32(art_count) * 100 + 1,
                                suffix='{} articles prceeded from {}'.format(art_ind, art_count))
    return article_inds


def ind2word(ind, vocab):
    return vocab[ind]


def inds2words(ind, vocab):
    return vocab[ind]


def generate_vocab(sentences, vocab_size=VOCAB_SIZE):
    print 'generate vocabulary'
    all_tokens = list(itertools.chain.from_iterable(i for i in sentences))
    word_freq = nltk.FreqDist(all_tokens)
    print "Found %d unique words tokens." % len(word_freq.items())
    VOCAB = np.array(zip(*word_freq.most_common(vocab_size))[0])
    VOCAB = np.delete(VOCAB, np.where(VOCAB == ART_END_TOK))
    VOCAB = np.append(VOCAB, UTOK)
#    VOCAB = np.append(VOCAB, ART_END_TOK)
    VOCAB = np.append(VOCAB, PTOK)
    print 'vocabulary with {0} tokens'.format(len(VOCAB))
    return VOCAB


def pre_process_context(context_data_file, vocab_size=VOCAB_SIZE, row_num=None):
#-----------------------------------------------------------load whoosh results
    _helpScripts.print_msg('load whoosh results')
    with open(XAPIAN_DIR + 'train.pickle', 'rb') as f:
        wh_pred_answ_train, wh_scores_train, wh_themes_train = pickle.load(f)
    with open(XAPIAN_DIR + 'test.pickle', 'rb') as f:
        wh_pred_answ_test, wh_scores_test, wh_themes_test = pickle.load(f)

    if row_num is not None:
        wh_pred_answ_train, wh_scores_train, wh_themes_train = \
            wh_pred_answ_train[:row_num], wh_scores_train[:row_num], wh_themes_train[:row_num]
        wh_pred_answ_test, wh_scores_test, wh_themes_test = \
            wh_pred_answ_test[:row_num], wh_scores_test[:row_num], wh_themes_test[:row_num]

    themes_train_ = [wh_th.flatten() for wh_th in wh_themes_train]
    themes_train = set([item for sublist in themes_train_ for item in sublist])

    themes_test_ = [wh_th.flatten() for wh_th in wh_themes_test]
    themes_test = set([item for sublist in themes_test_ for item in sublist])
    themes_all = themes_train | themes_test
    themes_all -= set([0])
    themes_name_all = set([os.path.splitext(os.path.basename(th))[0] for th in list(themes_all)])
    print '{} articles are used from whoosh results'.format(len(themes_name_all))
#--------------------------------------------------------------tokenize results
    _helpScripts.print_msg('tokenize results')
    sentences_, art_names = tokenize_docs(list(themes_all), mode='append',
                                          add_sep=False,
                                          max_sent_length=MAX_SENT_LENGTH,
                                          min_sent_length=MIN_SENT_LENGTH)
    sentences = []
    sentences.extend(list(itertools.chain(*sentences_)))
    sent_l = [len(s) for s in sentences]
    print '{} sentences with sentence length: min = {}, max = {}, median = {}'.format(len(sent_l), np.min(sent_l), np.max(sent_l), np.median(sent_l))
#-------------------------------------------------------------------make source
    _helpScripts.print_msg('make source')
    source = _aai_data.Source(lower=True, create_sents=False)
    sentences.extend(source.sent_train_tokens)
    sentences.extend(source.sent_test_tokens)
    quest_n_train = source.train_data.shape[0]
    quest_n_test = source.test_data.shape[0]
    print 'there are {} train questions and {} test questions'.format(quest_n_train, quest_n_test)
#-----------------------------------------------------------generate vocabulary
    _helpScripts.print_msg('generate vocabulary')
    VOCAB = generate_vocab(sentences, vocab_size=vocab_size)
    vocab_size = len(VOCAB)
    PAD_TOKEN_IND = len(VOCAB) - 1
#-------------------------------------------------------------generate indexies
    _helpScripts.print_msg('select themes')    
    _helpScripts.print_msg('for train', allign='left')
    themes_train = select_themes(wh_scores_train, wh_themes_train, themes_quest=ARTICLES_QUEST)
    themes_train_names = [[os.path.splitext(os.path.basename(th))[0] for th in th_q] for th_q in themes_train]
    _helpScripts.print_msg('for test', allign='left')
    themes_test = select_themes(wh_scores_test, wh_themes_test, themes_quest=ARTICLES_QUEST)
    themes_test_names = [[os.path.splitext(os.path.basename(th))[0] for th in th_q] for th_q in themes_test]
#-------------------------------------------------------------generate indexies
    _helpScripts.print_msg('generate indexies')
    _helpScripts.print_msg('for whole context', allign='left')

    sent_count = [[len(sents) for sents in sentences_[i]] for i in range(len(sentences_)) if art_names[i] in themes_all]
    sent_count = list(itertools.chain(*sent_count))
    
#    MAX_ART_THEME = np.median(sent_count)
    sentences_ = np.array(sentences_)
#    themes_all = list(themes_all)
    article_inds = get_art_word_ind(sentences_,
                                    VOCAB, MAX_SENT_ART,
                                    MAX_SENT_LENGTH, PAD_TOKEN_IND)
#-------------------------------------------------------------save data to file
    _helpScripts.print_msg('save data to file ' + context_data_file, allign='left')
    with open(context_data_file, 'wb') as f:
        cPickle.dump((VOCAB, sentences_, art_names, source,
                      article_inds,
                      themes_train, themes_test), f, protocol=2)
    _helpScripts.print_msg('completed')

    return (VOCAB, sentences_, art_names, source,
            article_inds, 
            themes_train, themes_test)


def pre_proc_quest(source, vocab, quest_data_file, row_num=None):
#-----------------------------------------------tokenize question / answer data
    _helpScripts.print_msg('generate question / answer data')

    quest_n_train = source.train_data.shape[0]
    quest_n_test = source.test_data.shape[0]

    answer_sym = ['A', 'B', 'C', 'D']
    answer_names = ['answer' + s for s in answer_sym]
    train_quest_tok = np.array([nltk.word_tokenize(sent) for sent in source.train_data['question'].values])
    test_quest_tok = np.array([nltk.word_tokenize(sent) for sent in source.test_data['question'].values])

    train_quest_sl = [len(t) for t in train_quest_tok]
    test_quest_sl = [len(t) for t in test_quest_tok]
    QUEST_SENT_L = np.max(train_quest_sl + test_quest_sl)

    train_answ_tok = np.transpose(np.array([[nltk.word_tokenize(sent) for sent in source.train_data[a].values] for a in answer_names]))
    test_answ_tok = np.transpose(np.array([[nltk.word_tokenize(sent) for sent in source.test_data[a].values] for a in answer_names]).reshape((-1, 4)))

    train_answ_sl = [len(t) for t in train_answ_tok.flatten()]
    test_answ_sl = [len(t) for t in test_answ_tok.flatten()]
    ANSW_SENT_L = np.max(train_answ_sl + test_answ_sl)
    print 'max question length is {}, answer length: {}'.format(QUEST_SENT_L, ANSW_SENT_L)
#-------------------------------------------------------generate the input data
    _helpScripts.print_msg('generate indexing for questions', allign='left')
    PAD_TOKEN_IND = len(vocab) - 1

    train_quest = np.expand_dims(int16(sents2indPar(train_quest_tok, vocab, proc=8, max_l=QUEST_SENT_L)), 1)
    test_quest = np.expand_dims(int16(sents2indPar(test_quest_tok, vocab, proc=8, max_l=QUEST_SENT_L)), 1)
    _helpScripts.print_msg('generate indexing for answers', allign='left')
    train_answ = int16(sents2indPar(train_answ_tok.reshape((-1)), vocab, proc=8, max_l=ANSW_SENT_L)).reshape((-1, 4, ANSW_SENT_L))
    test_answ = int16(sents2indPar(test_answ_tok.reshape((-1)), vocab, proc=8, max_l=ANSW_SENT_L)).reshape((-1, 4, ANSW_SENT_L))
    
    correct_answ_ = correct_answ=source.train_data['correctAnswer'].values
    train_correct_answ = float32([np.where(answ == np.array(answer_sym))[0][0] for answ in correct_answ_])
#-------------------------------------------------------------save data to file
    _helpScripts.print_msg('save data to file ' + quest_data_file, allign='left')
    with open(quest_data_file, 'wb') as f:
        cPickle.dump((train_quest, train_answ, train_correct_answ,
                      test_quest, test_answ), f, protocol=2)
    _helpScripts.print_msg('completed')

    return (train_quest, train_answ, train_correct_answ,
            test_quest, test_answ)


class BatchedDotLayer(LL.MergeLayer):
    """
    returns batched_dot product of two matrixies
    """

    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
#        return (input_shapes[0][0], input_shapes[0][1], input_shapes[1][2])
        return tuple(list(input_shapes[0][:-1]) + [input_shapes[1][-1]])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])


class CosineSimilLayer(LL.MergeLayer):
    def __init__(self, incomings, tol=1e-6, axis=1, **kwargs):
        super(CosineSimilLayer, self).__init__(incomings, **kwargs)
        self.tol = tol
        self.axis = axis
        
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 1)
        
    def get_output_for(self, inputs, **kwargs):
        mod1 = T.clip(T.sqrt(T.sum(T.sqr(inputs[0]), axis=self.axis)), self.tol, 1000.)
        mod2 = T.clip(T.sqrt(T.sum(T.sqr(inputs[1]), axis=self.axis)), self.tol, 1000.)
        return T.sum(T.mul(inputs[0], inputs[1]), axis=self.axis) / T.mul(mod1, mod2)


class TemporalEncodicgLayer(LL.Layer):
    def __init__(self, incoming, T, **kwargs):
        super(TemporalEncodicgLayer, self).__init__(incoming, **kwargs)
        self.T = self.add_param(T, self.input_shape[-2:], name="T")
#        self.T = T        
    
    def get_output_for(self, input):
        return input + self.T


class EncodingFullLayer(LL.MergeLayer):
    """
    performs complete encoding of the input
    """
    def __init__(self, incomings, vocab_size, emb_size, W, WT=None, **kwargs):
        super(EncodingFullLayer, self).__init__(incomings, **kwargs)
#        if len(self.input_shapes[0]) == 3:
#            batch_size, w_count, w_length = self.input_shapes[0]
        shape = tuple(self.input_shapes[0])
#        else:
#            shape = tuple(self.input_shapes[0])
        
        self.WT = None
#        self.reset_zero()
        self.l_in = LL.InputLayer(shape=shape)
        self.l_in_pe = LL.InputLayer(shape=shape + (emb_size,))
        self.l_emb = LL.EmbeddingLayer(self.l_in, input_size=vocab_size, output_size=emb_size, W=W)
        self.W = self.l_emb.W
        self.l_emb = LL.ElemwiseMergeLayer((self.l_emb, self.l_in_pe),
                                           merge_function=T.mul)
        self.l_emb_res = LL.ExpressionLayer(self.l_emb, lambda X: X.sum(2), output_shape='auto')
        
#        self.l_emb_res = SumLayer(self.l_emb, axis=2)
        if np.any(WT):
            self.l_emb_res = TemporalEncodicgLayer(self.l_emb_res, T=WT)
            self.WT = self.l_emb_res.T
        params = LL.helper.get_all_params(self.l_emb_res, trainable=True)
        values = LL.helper.get_all_param_values(self.l_emb_res, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(emb_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[-1, :], zero_vec_tensor)) for x in [self.W]])

    def reset_zero(self):
        self.set_zero(self.zero_vec)

    def get_output_shape_for(self, input_shapes):
        return LL.helper.get_output_shape(self.l_emb_res)

    def get_output_for(self, inputs, **kwargs):
        return LL.helper.get_output(self.l_emb_res, {self.l_in: inputs[0],
                                                     self.l_in_pe: inputs[1]})


class MemoryLayer(LL.MergeLayer):
    """
    memory layer from http://arxiv.org/pdf/1503.08895v4.pdf
    incomings: context raw, query embedded, (context_prob)
    """
    def __init__(self, incomings, vocab_size, emb_size,
                 A=lasagne.init.Normal(std=0.1), C=lasagne.init.Normal(std=0.1),
                 AT=lasagne.init.Normal(std=0.1), CT=lasagne.init.Normal(std=0.1),
                 nonlin=lasagne.nonlinearities.softmax,
                 RN=0., **kwargs):
        super(MemoryLayer, self).__init__(incomings, **kwargs)

        self.vocab_size, self.emb_size = vocab_size, emb_size
        self.nonlin = nonlin
        self.RN = RN
#        self.A, self.C, self.AT, self.CT = A, C, AT, CT

        batch_size, c_count, c_length = self.input_shapes[0]
        _, q_count, _ = self.input_shapes[2]

        self.l_c_in = LL.InputLayer(shape=(batch_size, c_count, c_length))
        self.l_c_in_pe = LL.InputLayer(shape=(batch_size, c_count, c_length, self.emb_size))
        self.l_u_in = LL.InputLayer(shape=(batch_size, q_count, self.emb_size))

        self.l_c_A_enc = EncodingFullLayer((self.l_c_in, self.l_c_in_pe), self.vocab_size, self.emb_size, A, AT)
        self.l_c_C_enc = EncodingFullLayer((self.l_c_in, self.l_c_in_pe), self.vocab_size, self.emb_size, C, CT)
        self.A, self.C = self.l_c_A_enc.W, self.l_c_C_enc.W
        self.AT, self.CT = self.l_c_A_enc.WT, self.l_c_C_enc.WT
        if len(incomings) == 4:        # if there is also the probabilities over sentences
            self.l_in_ac_prob = LL.InputLayer(shape=(batch_size, c_count, emb_size))
            self.l_c_A_enc_ = LL.ElemwiseMergeLayer((self.l_c_A_enc, self.l_in_ac_prob),
                                                    merge_function=T.mul)
            self.l_c_C_enc_ = LL.ElemwiseMergeLayer((self.l_c_C_enc, self.l_in_ac_prob),
                                                    merge_function=T.mul)
        
        self.l_u_in_tr = LL.DimshuffleLayer(self.l_u_in, pattern=(0, 2, 1))
        if len(incomings) == 4:
            self.l_p = BatchedDotLayer((self.l_c_A_enc_, self.l_u_in_tr))
        else:
            self.l_p = BatchedDotLayer((self.l_c_A_enc, self.l_u_in_tr))

        if self.l_p.output_shape[2]==1:
            self.l_p = LL.FlattenLayer(self.l_p, outdim=2)
#            self.l_p = LL.DimshuffleLayer(self.l_p, (0, 1))

        if self.nonlin == 'MaxOut':
            raise NotImplementedError
        self.l_p = LL.NonlinearityLayer(self.l_p, nonlinearity=nonlin)
        self.l_p = LL.DimshuffleLayer(self.l_p, (0, 1, 'x'))
#        self.l_p = LL.ReshapeLayer(self.l_p, self.l_p.output_shape + (1,))
        self.l_p = LL.ExpressionLayer(self.l_p, lambda X: X.repeat(emb_size, 2), output_shape='auto')
##        self.l_p = RepeatDimLayer(self.l_p, emb_size, axis=2)
        if len(incomings) == 4:
            self.l_pc = LL.ElemwiseMergeLayer((self.l_p, self.l_c_C_enc_), merge_function=T.mul)
        else:
            self.l_pc = LL.ElemwiseMergeLayer((self.l_p, self.l_c_C_enc), merge_function=T.mul)            
        self.l_o = LL.ExpressionLayer(self.l_pc, lambda X: X.sum(1), output_shape='auto')
#        self.l_o = SumLayer(self.l_pc, axis=1)
        self.l_o = LL.DimshuffleLayer(self.l_o, pattern=(0, 'x', 1))
        self.l_o_u = LL.ElemwiseMergeLayer((self.l_o, self.l_u_in), merge_function=T.add)

        params = LL.helper.get_all_params(self.l_o_u, trainable=True)
        values = LL.helper.get_all_param_values(self.l_o_u, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)


    def get_output_shape_for(self, input_shapes):
        return LL.helper.get_output_shape(self.l_o_u)

    def get_output_for(self, inputs, **kwargs):
        if len(inputs) == 3:
            return LL.helper.get_output(self.l_o_u,
                                        {self.l_c_in: inputs[0],
                                         self.l_c_in_pe: inputs[1],
                                         self.l_u_in: inputs[2]})
        else:
            return LL.helper.get_output(self.l_o_u,
                                        {self.l_c_in: inputs[0],
                                         self.l_c_in_pe: inputs[1],
                                         self.l_u_in: inputs[2],
                                         self.l_in_ac_prob: inputs[3]})

    def get_cont_prob(self):
        return self.l_p

    def reset_zero(self):
        self.l_c_A_enc.reset_zero()
        self.l_c_C_enc.reset_zero()


def make_memnn(vocab_size, cont_sl,
               cont_wl, quest_wl,
               answ_wl, rnn_size, rnn_type='LSTM', pool_size=4,
               answ_n=4, dence_l=[100], dropout=0.5,
               batch_size=16, emb_size=50, grad_clip=40, init_std=0.1,
               num_hops=3, rnn_style=False, nonlin=LN.softmax,
               init_W=None, rng=None, art_pool=4,
               lr=0.01, mom=0, updates=LU.adagrad, valid_indices=0.2,
               permute_answ=False, permute_cont=False):

    def select_rnn(x):
        return {
            'RNN': LL.RecurrentLayer,
            'LSTM': LL.LSTMLayer,
            'GRU': LL.GRULayer,
        }.get(x, LL.LSTMLayer)
    
#    dence = dence + [1]
    RNN = select_rnn(rnn_type)
#-----------------------------------------------------------------------weights
    tr_variables = {}
    tr_variables['WQ'] = theano.shared(init_std*np.random.randn(vocab_size, emb_size).astype('float32'))
    tr_variables['WA'] = theano.shared(init_std*np.random.randn(vocab_size, emb_size).astype('float32'))
    tr_variables['WC'] = theano.shared(init_std*np.random.randn(vocab_size, emb_size).astype('float32'))
    tr_variables['WTA'] = theano.shared(init_std*np.random.randn(cont_sl, emb_size).astype('float32'))
    tr_variables['WTC'] = theano.shared(init_std*np.random.randn(cont_sl, emb_size).astype('float32'))
    tr_variables['WAnsw'] = theano.shared(init_std*np.random.randn(vocab_size, emb_size).astype('float32'))

#------------------------------------------------------------------input layers
    layers = [
        (LL.InputLayer, {'name': 'l_in_q', 'shape': (batch_size, 1, quest_wl), 'input_var': T.itensor3('l_in_q_')}),
        (LL.InputLayer, {'name': 'l_in_a', 'shape': (batch_size, answ_n, answ_wl), 'input_var': T.itensor3('l_in_a_')}),
        (LL.InputLayer, {'name': 'l_in_q_pe', 'shape': (batch_size, 1, quest_wl, emb_size)}),
        (LL.InputLayer, {'name': 'l_in_a_pe', 'shape': (batch_size, answ_n, answ_wl, emb_size)}),
        (LL.InputLayer, {'name': 'l_in_cont', 'shape': (batch_size, cont_sl, cont_wl), 'input_var': T.itensor3('l_in_cont_')}),
        (LL.InputLayer, {'name': 'l_in_cont_pe', 'shape': (batch_size, cont_sl, cont_wl, emb_size)})
    ]
#------------------------------------------------------------------slice layers
#    l_qs = []
#    l_cas = []
    l_a_names = ['l_a_{}'.format(i) for i in range(answ_n)]
    l_a_pe_names = ['l_a_pe{}'.format(i) for i in range(answ_n)]
    for i in range(answ_n):
        layers.extend([(LL.SliceLayer, {'name': l_a_names[i], 'incoming': 'l_in_a',
                                        'indices': slice(i, i+1), 'axis': 1})])
    for i in range(answ_n):
        layers.extend([(LL.SliceLayer, {'name': l_a_pe_names[i], 'incoming': 'l_in_a_pe',
                                        'indices': slice(i, i+1), 'axis': 1})])
#------------------------------------------------------------------MEMNN layers
#question----------------------------------------------------------------------
    layers.extend([(EncodingFullLayer, {'name': 'l_emb_f_q', 'incomings': ('l_in_q', 'l_in_q_pe'),
                                        'vocab_size': vocab_size, 'emb_size': emb_size,
                                        'W': tr_variables['WQ'], 'WT': None})])

    l_mem_names = ['ls_mem_n2n_{}'.format(i) for i in range(num_hops)]

    layers.extend([(MemoryLayer, {'name': l_mem_names[0],
                                  'incomings': ('l_in_cont', 'l_in_cont_pe', 'l_emb_f_q'),
                                  'vocab_size': vocab_size, 'emb_size': emb_size,
                                  'A': tr_variables['WA'], 'C': tr_variables['WC'],
                                  'AT': tr_variables['WTA'], 'CT': tr_variables['WTC'], 'nonlin': nonlin})])
    for i in range(1, num_hops):
        if i%2:
            WC, WA = tr_variables['WA'], tr_variables['WC']
            WTC, WTA = tr_variables['WTA'], tr_variables['WTC']
        else:
            WA, WC = tr_variables['WA'], tr_variables['WC']
            WTA, WTC = tr_variables['WTA'], tr_variables['WTC']
        layers.extend([(MemoryLayer, {'name': l_mem_names[i],
                                      'incomings': ('l_in_cont', 'l_in_cont_pe', l_mem_names[i-1]),
                                      'vocab_size': vocab_size, 'emb_size': emb_size,
                                      'A': WA, 'C': WC, 'AT': WTA, 'CT': WTC, 'nonlin': nonlin})])
#answers-----------------------------------------------------------------------
    l_emb_f_a_names = ['l_emb_f_a{}'.format(i) for i in range(answ_n)]
    for i in range(answ_n):
        layers.extend([(EncodingFullLayer, {'name': l_emb_f_a_names[i], 'incomings': (l_a_names[i], l_a_pe_names[i]),
                                            'vocab_size': vocab_size, 'emb_size': emb_size,
                                            'W': tr_variables['WAnsw'], 'WT': None})])
#------------------------------------------------------------concatenate layers
    layers.extend([(LL.ConcatLayer, {'name': 'l_qma_concat',
                                     'incomings': l_mem_names + l_emb_f_a_names})])
#--------------------------------------------------------------------RNN layers
    layers.extend([(RNN, {'name': 'l_qa_rnn_f', 'incoming': 'l_qma_concat',
#                          'mask_input': 'l_qamask_concat',
                          'num_units': rnn_size,
                          'backwards': False, 'only_return_final': False,
                          'grad_clipping': grad_clip})])
    layers.extend([(RNN, {'name': 'l_qa_rnn_b', 'incoming': 'l_qma_concat',
#                          'mask_input': 'l_qamask_concat',
                          'num_units': rnn_size,
                          'backwards': True, 'only_return_final': False,
                          'grad_clipping': grad_clip})])

    layers.extend([(LL.SliceLayer, {'name': 'l_qa_rnn_f_sl', 'incoming': 'l_qa_rnn_f',
                                    'indices': slice(-answ_n, None), 'axis': 1})])
    layers.extend([(LL.SliceLayer, {'name': 'l_qa_rnn_b_sl', 'incoming': 'l_qa_rnn_b',
                                    'indices': slice(-answ_n, None), 'axis': 1})])

    layers.extend([(LL.ElemwiseMergeLayer, {'name': 'l_qa_rnn_conc',
                                            'incomings': ('l_qa_rnn_f_sl', 'l_qa_rnn_b_sl'),
                                            'merge_function': T.add})])
#-----------------------------------------------------------------pooling layer
#    layers.extend([(LL.DimshuffleLayer, {'name': 'l_qa_rnn_conc_',
#                                         'incoming': 'l_qa_rnn_conc', 'pattern': (0, 'x', 1)})])
    layers.extend([(LL.Pool1DLayer, {'name': 'l_qa_pool',
                                     'incoming': 'l_qa_rnn_conc',
                                     'pool_size': pool_size, 'mode': 'max'})])
#------------------------------------------------------------------dence layers
    l_dence_names = ['l_dence_{}'.format(i) for i, _ in enumerate(dence_l)]
    if dropout:
        layers.extend([(LL.DropoutLayer, {'name': 'l_dence_do', 'p': dropout})])
    for i, d in enumerate(dence_l):
        if i < len(dence_l) - 1:
            nonlin = LN.tanh
        else:
            nonlin = LN.softmax
        layers.extend([(LL.DenseLayer, {'name': l_dence_names[i], 'num_units': d,
                                        'nonlinearity': nonlin})])
        if i < len(dence_l) - 1 and dropout:
            layers.extend([(LL.DropoutLayer, {'name': l_dence_names[i] + 'do', 'p': dropout})])

    if isinstance(valid_indices, np.ndarray) or isinstance(valid_indices, list):
        train_split=TrainSplit_indices(valid_indices=valid_indices)
    else:
        train_split=TrainSplit(eval_size=valid_indices, stratify=False)

    if permute_answ or permute_cont:
        batch_iterator_train = PermIterator(batch_size, permute_answ, permute_cont)
    else:
        batch_iterator_train = BatchIterator(batch_size=batch_size)

    def loss(x, t):
        return LO.aggregate(LO.categorical_crossentropy(T.clip(x, 1e-6, 1. - 1e-6), t))
#        return LO.aggregate(LO.squared_error(T.clip(x, 1e-6, 1. - 1e-6), t))

    nnet = NeuralNet(
            y_tensor_type=T.ivector,
            layers=layers,
            update=updates,
            update_learning_rate=lr,
#            update_epsilon=1e-7,
            objective_loss_function=loss,
            regression=False,
            verbose=2,
            batch_iterator_train=batch_iterator_train,
            batch_iterator_test=BatchIterator(batch_size=batch_size/2),
#            batch_iterator_train=BatchIterator(batch_size=batch_size),
#            batch_iterator_test=BatchIterator(batch_size=batch_size),            
            #train_split=TrainSplit(eval_size=eval_size)
            train_split=train_split,
            on_batch_finished=[zero_memnn]
        )
    nnet.initialize()
    PrintLayerInfo()(nnet)
    return nnet


#class ZeroMemNN:
#    def __init__(self):
#        pass
#
#    def __call__(self, nn, train_history=None):
#        for layer in nn.get_all_layers():
#            if "reset_zero" in dir(layer):
#                layer.reset_zero()
def zero_memnn(nn, train_history=None):
    for layer in nn.get_all_layers():
        if "reset_zero" in dir(layer):
            layer.reset_zero()


class PermIterator(BatchIterator):
    def __init__(self, batch_size, permute_answ=True, permute_cont=True):
        super(PermIterator, self).__init__(batch_size)
        self.permute_answ = permute_answ
        self.permute_cont = permute_cont

    def transform(self, Xb, yb):
        a_in = Xb['l_in_a']
        cont_in = Xb['l_in_cont']
        if self.permute_answ:
            a_new_idx = np.random.permutation(a_in.shape[1])
        else:
            a_new_idx = np.arange(a_in.shape[1])
        if self.permute_cont:
            c_new_ind = np.random.permutation(cont_in.shape[1])
        else:
            c_new_ind = np.arange(a_in.shape[1])
        X_new = {'l_in_q': Xb['l_in_q'], 'l_in_a': a_in[:, a_new_idx, :],
                 'l_in_q_pe': Xb['l_in_q_pe'], 'l_in_a_pe': Xb['l_in_a_pe'],
                 'l_in_cont': cont_in[:, c_new_ind, :], 'l_in_cont_pe': Xb['l_in_cont_pe']}

        y_new = int32([np.where(yb[i] == a_new_idx)[0][0] for i in range(yb.shape[0])])
        return X_new, y_new


def positional_encoding(J, sl, emb_size, samples):
    pe = np.zeros((1, 1, J, emb_size))
    for k in range(emb_size):
        for j in range(J):
            pe[0][0][j][k] = (1. - float32(j) / J) - (float32(k) / emb_size) * (1. - 2.*float32(j) / J)
    return np.repeat(np.repeat(pe, samples, axis=0), sl, axis=1)


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


if __name__ == "__main__":
    _helpScripts.print_msg('start at time ' + strftime("%Y-%m-%d %H:%M:%S"))
#    print strftime("%Y-%m-%d %H:%M:%S")
    startTime = datetime.datetime.now()
    MAX_SAMPLES = None
    os.chdir('/Users/lexx/Documents/Work/Kaggle/AllenAIScience/')
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
#--------------------------------------------------generate / load context data
    context_data_file = saveDir + 'context_data.cpickle'
    if os.path.isfile(context_data_file):
        _helpScripts.print_msg('load context data from file: ' + context_data_file)
        with open(context_data_file, 'rb') as f:
            (VOCAB, sentences_, art_names, source,
             article_inds, themes_train, themes_test) = cPickle.load(f)
    else:
        (VOCAB, sentences_, art_names, source,
         article_inds, themes_train, themes_test) = pre_process_context(context_data_file, vocab_size=VOCAB_SIZE, row_num=MAX_SAMPLES)
#    MAX_ART_THEME = theme_art_ind.shape[1]
    VOCAB_LENTH = len(VOCAB)
    PAD_TOKEN_IND = len(VOCAB) - 1
#-------------------------------------------------generate / load question data
    quest_data_file = saveDir + 'qa_data.cpickle'
    if os.path.isfile(quest_data_file):
        _helpScripts.print_msg('load qa data from file: ' + quest_data_file)
        with open(quest_data_file, 'rb') as f:
            (train_quest, train_answ, train_correct_answ, \
             test_quest, test_answ) = cPickle.load(f)
    else:
        (train_quest, train_answ, train_correct_answ, \
         test_quest, test_answ) = pre_proc_quest(source, VOCAB, quest_data_file)
    QUEST_SENT_L = train_quest.shape[-1]
    ANSW_SENT_L = train_answ.shape[-1]
    if not MAX_SAMPLES:
        quest_n_train = train_quest.shape[0]
        quest_n_test = test_quest.shape[0]
    else:
        quest_n_train = MAX_SAMPLES
        quest_n_test = MAX_SAMPLES
#    targ_lb = LabelBinarizer()
#    train_targets = targ_lb.fit_transform(int32(train_correct_answ))
    train_targets = int32(train_correct_answ)
#----------------------------------------rearrange indexies regarding questions
    _helpScripts.print_msg('rearrange indexies regarding questions')
    context_train = np.ones((quest_n_train, ARTICLES_QUEST,
                             MAX_SENT_ART, MAX_SENT_LENGTH), dtype='int16') * PAD_TOKEN_IND
    context_test = np.ones((quest_n_test, ARTICLES_QUEST,
                             MAX_SENT_ART, MAX_SENT_LENGTH), dtype='int16') * PAD_TOKEN_IND
    art_names_dict = dictionary = dict(zip(art_names, range(len(art_names))))

    art_train_ind = np.array([[art_names_dict[art]
                              for art in art_q] for art_q in themes_train])
    art_test_ind = np.array([[art_names_dict[art]
                             for art in art_q] for art_q in themes_test])

    _helpScripts.print_msg('for train', allign='left')
    for i in range(quest_n_train):
        context_train[i] = article_inds[art_train_ind[i].ravel()]
    for i in range(quest_n_test):
        context_test[i] = article_inds[art_test_ind[i].ravel()]
    context_train = np.reshape(context_train, (quest_n_train, ARTICLES_QUEST*MAX_SENT_ART, -1))
    context_test = np.reshape(context_test, (quest_n_test, ARTICLES_QUEST*MAX_SENT_ART, -1))
#----------------------------------------------------------------generate MEMNN
    EMB_SIZE = 50
    BATCH_SIZE = 16
    NUM_HOPS = 3
#    ART_SEL = 4
#    ARTICLES_QUEST_ = 4
    rnn_type = 'LSTM'
    rnn_size = 20
#    dropout = 0.3
    train_indices = np.array(source.trainInd)
    valid_indices = np.array(source.testInd)
    BATCH_SIZE = 8
    N_EPOCHS = 100
#    NUM_HOPS = 3
#    ART_SEL = 4
#    ARTICLES_QUEST_ = 5
    lr = 0.01
    dence = [20, 4]

    q_pe_train = positional_encoding(QUEST_SENT_L, 1, EMB_SIZE, quest_n_train)
    a_pe_train = positional_encoding(ANSW_SENT_L, 4, EMB_SIZE, quest_n_train)
    cont_pe_train = positional_encoding(MAX_SENT_LENGTH, ARTICLES_QUEST*MAX_SENT_ART, EMB_SIZE, quest_n_train)

    _helpScripts.print_msg('create memory NN')
    print 'batch_size: {}, n_epochs: {}'.format(BATCH_SIZE, N_EPOCHS)
    print 'emb_size: {}, num_hops: {}'.format(EMB_SIZE, NUM_HOPS)
    net = make_memnn(vocab_size=len(VOCAB), cont_sl=ARTICLES_QUEST*MAX_SENT_ART,
                     cont_wl=MAX_SENT_LENGTH, quest_wl=QUEST_SENT_L,
                     answ_wl=ANSW_SENT_L, rnn_size=rnn_size, rnn_type='LSTM', pool_size=4,
                     answ_n=4, dence_l=dence, dropout=0.5,
                     batch_size=BATCH_SIZE, emb_size=EMB_SIZE, grad_clip=40, init_std=0.1,
                     num_hops=3, rnn_style=False, nonlin=LN.softmax,
                     init_W=None, rng=None, art_pool=4,
                     lr=0.01, mom=0, updates=LU.adagrad, valid_indices=0.2,
                     permute_answ=True, permute_cont=True)

#----------------------------------------------------------------------train NN
    _helpScripts.print_msg('train NN')
    net.fit({'l_in_q': train_quest[:quest_n_train], 'l_in_a': train_answ[:quest_n_train],
             'l_in_q_pe': float32(q_pe_train), 'l_in_a_pe': float32(a_pe_train),
             'l_in_cont': int32(context_train), 'l_in_cont_pe': float32(cont_pe_train)},
            train_targets[:quest_n_train], epochs=N_EPOCHS)

    plot_loss(net)

#-----------------------------------------------------------------------save NN
    net.save_params_to(saveDir + 'params.picle')
#    net.save_weights_to(saveDir + 'weights.picle')
    print strftime("%Y-%m-%d %H:%M:%S")
    print "completed in ", datetime.datetime.now() - startTime
#--------------------------------------------------------------predict for test
    q_pe_test = positional_encoding(QUEST_SENT_L, 1, EMB_SIZE, quest_n_test)
    a_pe_test = positional_encoding(ANSW_SENT_L, 4, EMB_SIZE, quest_n_test)
    cont_pe_test = positional_encoding(MAX_SENT_LENGTH, ARTICLES_QUEST*MAX_SENT_ART, EMB_SIZE, quest_n_test)
    pred_answ_test = net.predict({'l_in_q': test_quest[:quest_n_test], 'l_in_a': train_answ[:quest_n_test],
                                  'l_in_q_pe': float32(q_pe_test), 'l_in_a_pe': float32(a_pe_test),
                                  'l_in_cont': int32(context_test), 'l_in_cont_pe': float32(cont_pe_test)})

#------------------------------------------------------------------------submit
    if SUBMIT:
        submit_file_name = "/Users/lexx/Documents/Work/Kaggle/AllenAIScience/submissions/subm_memnn_{0}.csv".format(VERSION)
        submit_metafDF = pd.DataFrame(list(pred_answ_test)[0:len(source.test_data.index)],
                                      index=source.test_data.index,
                                      columns=['correctAnswer'])
        submit_metafDF.to_csv(submit_file_name)