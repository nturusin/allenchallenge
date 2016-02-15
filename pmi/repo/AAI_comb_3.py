# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:16:43 2016

@author: lexx
"""

import numpy as np
import pickle
import cPickle

import pandas as pd

# sys.path.append("/Users/lexx/Documents/Work/python/")
# sys.path.append("/Users/lexx/Documents/Work/Kaggle/AllenAIScience/python/")
# sys.path.append("/Users/lexx/Documents/Work/python/Development/")
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

import AAI_data_3 as _aai_data
import help_scripts as _helpScripts
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

VERSION = 4
PMI_VERSION = 15
XAPIAN_VERSION = 5

SUBMIT = True


def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1).reshape((-1, 1))

    return dist


def combine_scores_x(scores, mode='sum', coeff=1):
    scores_ = np.asanyarray(scores)
    if mode == 'sum':
        scores_ = scores_ * coeff
        scores__ = np.sum(scores_, 2)
    else:
        scores__ = scores_[:, :, 0]
    return scores__


def predict_answer(answer_sym, scores):
    pred_answ_ = np.argmax(scores, axis=1)
    pred_answ = [answer_sym[i] for i in pred_answ_]
    return pred_answ


if __name__ == "__main__":
    # os.chdir('/Users/lexx/Documents/Work/Kaggle/AllenAIScience/')
    #    corpus_tokens = get_all_corpus(search_dir)
    xapian_dir = './data/xapian/{0}/'.format(XAPIAN_VERSION)
    pmi_dir = './data/pmi/{0}/'.format(PMI_VERSION)
    # ------------------------------------------------------------------load results
    _helpScripts.print_msg('load results')
    _helpScripts.print_msg('pmi version {}'.format(PMI_VERSION), allign='left')
    with open(pmi_dir + 'train.pickle', 'rb') as f:
        (pmi_pred_answ_train, pmi_train, pmi_colls_train) = cPickle.load(f)
    with open(pmi_dir + 'test.pickle', 'rb') as f:
        (pmi_pred_answ_test, pmi_test, pmi_colls_test) = cPickle.load(f)

    _helpScripts.print_msg('xapian version {}'.format(XAPIAN_VERSION), allign='left')
    with open(xapian_dir + 'train.pickle', 'rb') as f:
        (x_pred_answ_train, x_train_scores, x_train_paths) = pickle.load(f)
    with open(xapian_dir + 'test.pickle', 'rb') as f:
        (x_pred_answ_test, x_test_scores, x_test_paths) = pickle.load(f)
    # ------------------------------------------------------------------------source
    source = _aai_data.Source(create_sents=False)
    # ---------------------------------------------------------------combine results
    x_train_scores_ = combine_scores_x(x_train_scores, mode='sum',
                                       coeff=np.exp(np.arange(0.1, 4, (4 - 0.1) / x_train_scores.shape[-1]))[::-1])
    x_test_scores_ = combine_scores_x(x_test_scores, mode='sum',
                                      coeff=np.exp(np.arange(0.1, 4, (4 - 0.1) / x_test_scores.shape[-1]))[::-1])
    _helpScripts.print_msg('predict for train')

    x_ssc = StandardScaler()
    pmi_ssc = StandardScaler()

    x_train_scores_sc = np.zeros_like(x_train_scores_)
    x_train_scores_sc[source.trainInd] = x_ssc.fit_transform(x_train_scores_[source.trainInd])
    x_train_scores_sc[source.testInd] = x_ssc.transform(x_train_scores_[source.testInd])
    x_test_scores_sc = x_ssc.transform(x_test_scores_)

    pmi_train_scores_sc = np.zeros_like(pmi_train)
    pmi_train_scores_sc[source.trainInd] = pmi_ssc.fit_transform(pmi_train[source.trainInd])
    pmi_train_scores_sc[source.testInd] = pmi_ssc.transform(pmi_train[source.testInd])
    pmi_test_scores_sc = pmi_ssc.transform(pmi_test)

    # x_train_scores_sc = (x_train_scores_- np.mean(x_train_scores_, axis=1).
    # reshape((-1, 1))) / np.std(x_train_scores_, axis=1).reshape((-1, 1))
    # pmi_train_scores_sc = (pmi_train - np.mean(pmi_train, axis=1).reshape((-1, 1))) / np.std(pmi_train, axis=1).
    # reshape((-1, 1))

    model_features = np.concatenate((x_train_scores_sc, pmi_train_scores_sc), axis=1)
    model_features_test = np.concatenate((x_test_scores_sc, pmi_test_scores_sc), axis=1)

    #    model_features = x_train_scores_sc
    #    model = SVC(C=1., verbose=True, kernel='linear')

    model = LogisticRegression(C=4.6415888336127775, class_weight=None, dual=False,
                               fit_intercept=True, intercept_scaling=1, max_iter=50,
                               multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
                               solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

    model.fit(model_features[source.trainInd], source.train_data['correctAnswer'].values[source.trainInd])

    # run randomized search
    # clf = LogisticRegression()
    # c_range = np.logspace(-2, 6, 10)
    # penalty = ['l1', 'l2']
    # max_iter = range(50, 5000, 50)
    # rsearch = RandomizedSearchCV(clf, param_distributions=dict(C=c_range, penalty=penalty, max_iter=max_iter), n_iter=1980)
    # rsearch.fit(model_features, source.train_data['correctAnswer'].values)
    # print(rsearch)
    # print(rsearch.best_score_)
    # print(rsearch.best_estimator_)
    # print(rsearch.best_params_)

    # X_train = model_features[source.trainInd]
    # y_train = source.train_data['correctAnswer'].values[source.trainInd]
    #
    # X_test = model_features[source.testInd]
    # y_test = source.train_data['correctAnswer'].values[source.testInd]

    pred_answ_train = model.predict(model_features)

    #    pred_answ_train = [source.answer_sym[i] for i in np.argmax(train_res_comb, axis=1)]
    accur = accuracy_score(pred_answ_train[source.testInd], source.train_data['correctAnswer'].values[source.testInd])
    accur_tr = accuracy_score(pred_answ_train[source.trainInd],
                              source.train_data['correctAnswer'].values[source.trainInd])
    print 'accuracy', accur_tr, accur

    _helpScripts.print_msg('predict for test')
    pred_answ_test = model.predict(model_features_test)

    # ------------------------------------------------------------------save results

    if SUBMIT:
        submit_file_name = "./submissions/subm_x_pmi_comb_{0}.csv".format(VERSION)
        submit_metafDF = pd.DataFrame(list(pred_answ_test)[0:len(source.test_data.index)],
                                      index=source.test_data.index,
                                      columns=['correctAnswer'])
        submit_metafDF.to_csv(submit_file_name)
