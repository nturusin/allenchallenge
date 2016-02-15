# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:49:32 2015

@author: lexx
"""


import os
import sys
import pickle
import traceback
import itertools
import numpy as np
import pandas as pd
import itertools
import glob
import time
from joblib import Parallel, delayed 

import random

import nltk
#import whoosh
#import string
from gensim.models import Word2Vec, Phrases
import logging
from nltk.tokenize import RegexpTokenizer

#sys.path.append("/Users/lexx/Documents/Work/python/")
#sys.path.append("/Users/lexx/Documents/Work/Kaggle/AllenAIScience/python/")
import AAI_data_3 as _aai_data
import help_scripts as _helpScripts


VERSION = 13
SUBMIT = True

def float32(k):
    return np.cast['float32'](k)
    
def sent_tolken_reg(sents):
    """
    http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    """
    tokenizer = RegexpTokenizer(r'\w+')
    x = []
    
    if isinstance(sents, list):
        for sent in sents:
            s_temp = []
            if isinstance(sent, list):
                for i in range(len(sent)):
                    s_temp.append(sent[i])
                x.extend(tokenizer.tokenize(s) for s in s_temp)
            else:
                x.extend(tokenizer.tokenize(sent))
    else:
        x = tokenizer.tokenize(sents)
    return x


def predict_semil(model, questions, answers, bt=None):
    """
    predicts the answer regarding the similiarity in word2vek model
    between question and answer
    """
#    tokenizer = RegexpTokenizer(r'\w+')
    
    quest_num = len(questions)
    pred_answ = np.repeat('', quest_num)
    simil = np.zeros((quest_num, 4), dtype='float32')
    for r in range(quest_num):
#        simil = np.zeros(4, dtype='float32')
        for i in range(4):
            print answers[r][i].split()
            try:
#                print nltk.word_tokenize(questions[r])
#                print nltk.word_tokenize(answers[r][i])
#                print simil
#                answ = answers[r][i].split()
                if not bt:
                    simil[r, i] = model.n_similarity(nltk.word_tokenize(questions[r]),
                                                     nltk.word_tokenize(answers[r][i]))
                else:
                    simil[r, i] = model.n_similarity(bt[nltk.word_tokenize(questions[r])],
                                                     bt[nltk.word_tokenize(answers[r][i])])
#                print simil
            except KeyError:
                print traceback.print_exception()
#                continue
        pred_answ[r] = answer_sym[np.argmax(simil[r, :])]
        
    return pred_answ, simil


class MySentences(object):
    """
    modified from http://rare-technologies.com/word2vec-tutorial/
    """
    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
#        for fname in os.listdir(self.dirname):
        for line in open(self.fileName):
            yield line.split()


def tokenize_doc(fileName):
    with open (fileName, "r") as myfile:
        text = myfile.read()
    text = text.decode('utf-8').lower()
    sentences = nltk.sent_tokenize(text)
    return [nltk.word_tokenize(sent) for sent in sentences]


def tokenize_docs(dirName):
    files = glob.glob(dirName + '*.txt')
    token_sent = []
    doc_sents = Parallel(n_jobs=8, verbose=2)(delayed(tokenize_doc)(f)
                        for f in files)
#    for i, f in enumerate(files[:]):
#        token_sent.extend(tokenize_doc(f))
#        _helpScripts.print_perc(float32(i)/float32(len(files)) * 100 + 1)
#    for ds in doc_sents:
    for ds in doc_sents:
        token_sent.extend(ds)
    return token_sent


def calc_accur(pred_answ, correct_answ):
    correct_answ = [1 if pred == gt else 0
                    for (pred, gt) in zip(pred_answ,
                                          correct_answ)]
    
    return sum(float32(correct_answ) / len(correct_answ))
    

def w2vec_train(train_model, sentences, bt=None, epoch_num=10,
                alpha=0.1, min_alpha=0.001,
                eval_quest=None, eval_answ=None, correct_answ=None):
    
#    print eval_quest
    random.seed(156)    
    alpha_delta = (alpha - min_alpha) / epoch_num
    alpha_res = [0 for i in range(epoch_num)]
    accur_res = [0 for i in range(epoch_num)]
    
    for epoch in range(epoch_num):
        alpha_res[epoch] = alpha
        print 'epoch {0}, alpha {1}\n'.format(epoch, alpha)
        random.shuffle(sentences)  # shuffling gets best results
        
        train_model.alpha, train_model.min_alpha = alpha, alpha
        if bt:
            train_model.train(bt[sentences])
        else:
            train_model.train(sentences)
        
        alpha = alpha - alpha_delta
        
        if eval_quest is not None:
            (pred_answ, simil_train) = predict_semil(w2vec_model, eval_quest,
                                                     eval_answ, bt)
            accur = calc_accur(pred_answ, correct_answ)
            print 'accuracy', accur
            accur_res[epoch] = accur
            time.sleep(5)
    return (train_model, accur_res, alpha_res)


def create_submission(w2vec_model, source):
    w2vec_model.init_sims(replace=True)
    print 'predicting with similiarity'

#    pred_answ = np.repeat('', quest_num_train)    
    print 'predict for train'
    (pred_answ_train, simil_train) = predict_semil(w2vec_model,
                                     source.train_data['question'].values,
                                     source.train_data[source.answer_names].values, None)
    print 'pred_answ_train'
    print pred_answ_train
    print 'simil_train'
    print simil_train

    for i in range(4):
        print answer_sym[i], ': ', sum(pred_answ_train == answer_sym[i])
    
#    correct_answ = [1 if pred == gt else 0
#                    for (pred, gt) in zip(pred_answ_train,
#                                          source.train_data['correctAnswer'])]
    
#    accur = sum(float32(correct_answ) / quest_num_train)
    accur = calc_accur(pred_answ_train, source.train_data['correctAnswer'])
    
    print 'accuracy', accur    
    
    print 'predict for test'
    (pred_answ_test, simil_test) = predict_semil(w2vec_model,
                                   source.test_data['question'].values,
                                   source.test_data[source.answer_names].values, None)
    print pred_answ_test
    for i in range(4):
        print answer_sym[i], ': ', sum(pred_answ_test == answer_sym[i])
    
    print 'save results'
    train_res_file = open(saveDir + 'train.pickle', 'wb')
    test_res_file =  open(saveDir + 'test.pickle', 'wb')
    pickle.dump((pred_answ_train, simil_train), train_res_file)
    pickle.dump((pred_answ_test, simil_test), test_res_file)
    train_res_file.close()
    test_res_file.close()
    
    
    if SUBMIT:
        submit_file_name = "./submissions/subm_w2v_{0}.csv".format(VERSION)
        submit_metafDF = pd.DataFrame(pred_answ_test[0:len(source.test_data.index)],
                                      index=source.test_data.index,
                                      columns=['correctAnswer'])
        submit_metafDF.to_csv(submit_file_name)


if __name__ == "__main__":
    #os.chdir('/Users/lexx/Documents/Work/Kaggle/AllenAIScience/')
    sent_file = './data/w2v_models/sentences.pickle'
    saveDir = './data/w2v_models/{0}/'.format(VERSION)
    w2vec_model_name = saveDir + 'm{0}'.format(VERSION)
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    source = _aai_data.Source()

    # train word2vec on the two sentences
#    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='./data/w2v_models/mylog.log', mode='a')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    answer_sym = ['A', 'B', 'C', 'D']
    answer_names = ['answer' + s for s in answer_sym]

#    quest_num_train = source.train_data.shape[0]


    if not os.path.isfile(w2vec_model_name):
        _helpScripts.print_msg('prepare the model')        
        w2vec_model = Word2Vec(size=500, alpha=0.1, window=5, min_count=5,
                               max_vocab_size=None, sample=1e-5, seed=1, workers=16,
                               min_alpha=0.0001, sg=1, hs=1, negative=5,
                               cbow_mean=0, iter=2, null_word=0, trim_rule=None)

#        if os.path.isfile('./data/w2v_models/sentences'):
#            with open(sent_file, 'rb') as f:
#                sentences = pickle.load(f)
#        else:

        wiki_dir = './data/corpus/wiki_text_mod/'
        ck12_full_spl_dir = u'./data/corpus/ck12_full_themes_mod/'
        ck12_wiki_spl_dir = u'./data/corpus/wiki_text_ck12_themes_mod/'

        _helpScripts.print_msg('tokenize wiki', allign='left')
        sentences_wiki  = tokenize_docs(wiki_dir)
        _helpScripts.print_msg('tokenize ck_12', allign='left')
        sentences_ck_12 = tokenize_docs(ck12_full_spl_dir)
        _helpScripts.print_msg('tokenize ck_12 wiki', allign='left')
        sentences_ck_12_wiki = tokenize_docs(ck12_wiki_spl_dir)

        sentences = list(sentences_wiki)
        sentences.extend(sentences_ck_12)
        sentences.extend(sentences_ck_12_wiki)
#            with open(sent_file, 'wb') as f:
#                pickle.dump(sentences, f, 6)
        
#        bt = Phrases(sentences, threshold=50)
#        bt.save(w2vec_model_name + '_bt')
        
        _helpScripts.print_msg('calculate vocabulary', allign='left')
        w2vec_model.build_vocab(sentences)
#        print 'training'
        _helpScripts.print_msg('train')        
        w2vec_model.train(sentences)
#        (w2vec_model, accur, alpha) = \
#                w2vec_train(w2vec_model, sentences, bt=None, epoch_num=10,
#                            alpha=0.03, min_alpha=0.0001)

#        w2vec_model.train(bt[sentences])
        _helpScripts.print_msg('save w2v to {}'.format(w2vec_model_name))
        w2vec_model.save(w2vec_model_name)
    else:
        _helpScripts.print_msg('load w2v from {}'.format(w2vec_model_name))
        w2vec_model = Word2Vec.load(w2vec_model_name)
    
    create_submission(w2vec_model, source)
