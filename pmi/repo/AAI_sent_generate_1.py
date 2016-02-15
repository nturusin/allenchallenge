# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:41:03 2015

@author: lexx
data retrieving
generates new sentences and from context and searches for context
"""

import sys
import os
import copy
import numpy as np
import glob
import whoosh
from whoosh import scoring as _wh_scoring
from whoosh import qparser as _wh_qparser

from datetime import datetime, timedelta

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
import pycountry

#from time import strftime
from dateutil.parser import parse as date_parse
from joblib import Parallel, delayed

sys.path.append("/Users/lexx/Documents/Work/python/")
sys.path.append("/Users/lexx/Documents/Work/Kaggle/AllenAIScience/python/")
sys.path.append("/Users/lexx/Documents/Work/python/Development/")
import AAI_data_3 as _aai_data
import help_scripts as _helpScripts
import AAI_quest_rewrite_3 as _questRewr

import nltk

SUBMIT = True

WHOOSH_VERSION = 14
WHOOSH_DIR = './data/corpus/index/{0}/'.format(WHOOSH_VERSION)

MIN_SENT_N = 1
MAX_SENT_N = 5

VERSION = 2

def get_country_names():
    countries = list(pycountry.countries)
    countries_names = [c.name.lower() for c in countries]
    return countries_names


def is_country(word, countries_names=get_country_names()):
    return word.lower().replace(' ', '') in countries_names


def select_sents(art_text, nlp, min_sent_n=1, max_sent_n=5,
                 max_sent_l=30, rng=np.random.RandomState(156), max_iter=20):
    sent_offs = 3
    sents = [unicode(s) for s in nltk.sent_tokenize(art_text)]
    sents_docs = [nlp(s) for s in sents]
    sents_n = len(sents)
    all_sents = False
    i_ = 0
    try:
        while not all_sents and i_ < max_iter:
            sent_n = rng.randint(min_sent_n, max_sent_n+1)
            sent_i0 = rng.randint(sent_offs, sents_n - sent_offs - sent_n)
            sent_in = sent_i0 + sent_n - 1
            if len(nltk.word_tokenize(sents[sent_in])) < max_sent_l:
                all_sents = np.all([_questRewr.is_compl_sentence(s)
                                    for s in sents_docs[slice(sent_i0, sent_in)]])
            i_+=1
    except ValueError:
        return None
    if all_sents:
        return sents[slice(sent_i0, sent_in)]
    else:
        return None


def replace_tok_to_spans(subtrees, noun_chunks):
    nc_subtrees = []
    for sti, st in enumerate(subtrees):
        for nci, nc in enumerate(noun_chunks):
            if np.all([nc_ in st for nc_ in nc]):
                nc_idx = np.where([nc[0] == st_ for st_ in st])[0][0]
                nc_subtrees.append((nci, sti, slice(nc_idx, nc_idx+len(nc))))
    subtrees_ = list(subtrees)
    nc_subtrees = sorted(nc_subtrees, key=lambda x: (x[1], x[2]), reverse=True)
    for nci, sti, nc_idx in nc_subtrees:
#        sti, nc_idx = nc_subtrees[nci]
        del subtrees_[sti][nc_idx]
        subtrees_[sti].insert(nc_idx.start, noun_chunks[nci])
    return subtrees_


def combine_nc_se(noun_chunks, sent_ents):
    rem_nc = []
    for ni, nc in enumerate(noun_chunks):
        if np.any([np.any([nc_ in se for nc_ in nc]) for se in sent_ents]):
            rem_nc += [ni]
    for ni in sorted(rem_nc)[::-1]:
        del noun_chunks[ni]
    for se in sent_ents:
        if se in noun_chunks:
            noun_chunks[np.where([se==noun_chunks[i] for i in range(len(noun_chunks))])[0][0]] = se
        else:
            noun_chunks += [se]
    return noun_chunks


def get_root_sent(sent_doc):
    for wi, w in enumerate(sent_doc):
        if w.dep_ == 'ROOT':
            return wi


def get_not_pos(sent_doc):
    for wi, w in enumerate(sent_doc):
        if w.dep_ == 'ROOT':
            if w.lemma_ == 'be' or wi == 0:
                return wi + 1
            else:
                return wi


def get_subtrees_pos(sent_doc, subtrees_):
    sent_doc_ = list(sent_doc)
    st_start = [sent_doc_.index(st[0]) if st[0].__class__.__name__ == 'Token'
                else sent_doc_.index(st[0][0]) for st in subtrees_]
    st_end = [sent_doc_.index(st[-1]) if st[-1].__class__.__name__ == 'Token'
              else sent_doc_.index(st[-1][-1]) for st in subtrees_]

    return zip(st_start, st_end)


def get_sent_pos(sent_doc, word):
    if word.__class__.__name__ == 'Token':
        return (list(sent_doc).index(word), list(sent_doc).index(word))
    if word.__class__.__name__ == 'Span':
        return (list(sent_doc).index(word[0]), list(sent_doc).index(word[-1]))


def get_sent_mod(sents, nlp, num=5, rng=np.random.RandomState(156)):
    """
    modifies the sentence by changing the mode word to opposite
    """
    if not sents:
        return None
    add_sents = _helpScripts.flatten_str_list([nltk.word_tokenize(sent) for sent in sents[:-1]])
    sents_true = []
    sents_false = []
#    modes = np.asanyarray(['num', 'date', 'place', 'name', 'adj', 'noun', 'neg'])
    sent_last = sents[-1]
    sent_doc = nlp(sent_last)
    not_pos = get_not_pos(sent_doc)
#    modes_ = modes[rng.permutation(len(modes))]
    subtrees = [list(d.subtree) for d in sent_doc if d.dep_!='ROOT' and d.head.dep_=='ROOT']
    noun_chunks = list(sent_doc.noun_chunks)
    sent_ents = list(sent_doc.ents)

    noun_chunks = combine_nc_se(noun_chunks, sent_ents)

    subtrees_ = replace_tok_to_spans(subtrees, noun_chunks)
    pos_ = [[s.pos_ if s.__class__.__name__=='Token' else s.root.pos_ for s in s_]
            for s_ in subtrees_]
    subtrees_ = [s for i, s in enumerate(subtrees_) if len(s) > 1 or pos_[i][0]=='NOUN']
    labels_ = [[None if s.__class__.__name__=='Token' else s.label_ for s in s_]
               for s_ in subtrees_]
    tag_ = [[s.tag_ if s.__class__.__name__=='Token' else s.root.tag_ for s in s_]
            for s_ in subtrees_]
    dep_ = [[s.dep_ if s.__class__.__name__=='Token' else s.root.dep_ for s in s_]
            for s_ in subtrees_]
#    conj_pos = get_sent_conj_pos(dep_)
#    if conj_pos:
#        sents_true, sents_false = change_conj(sent_doc, subtrees_, conj_pos, num=num)
#    else:        
    sent_key = get_keys(labels_)
    if sent_key:
        sents_true, sents_false = change_key(sent_doc, subtrees_, sent_key, num=num, rng=rng)
    if not sents_true:
        sents_true = [[w.orth_ for w in sent_doc]]
    sents_false += change_neg(sents_true, not_pos)
    try:
        _, sents_false_ = change_antonym(sent_doc, num=num, rng=rng)
        if sents_false_:
            sents_false += sents_false_
    except WordNetError:
        pass
    sents_true_f = [add_sents + sent for sent in sents_true]
    sents_false_f = [add_sents + sent for sent in sents_false]
    return sents_true_f, sents_false_f


def get_sent_conj_pos(dep_):
    conj_pos = {}
    for sti, st in enumerate(dep_):
        cpos_ = []
        for wi, w in enumerate(st):
            if w == 'conj':
                cpos_ += [wi]
        if len(cpos_) > 1:
            conj_pos[sti] = cpos_
    return conj_pos


def change_conj(sent_doc, subtrees_, conj_pos, num=5,
                rng=np.random.RandomState(156)):
    sents_true = []
    sents_false = []
    root_pos = get_root_sent(sent_doc)

    st_conj = 0
    conjs = np.array(subtrees_[conj_pos.items()[st_conj][0]])\
                              [conj_pos.items()[st_conj][1]]
    conj_poss = (get_sent_pos(sent_doc, conjs[0])[0],
                 get_sent_pos(sent_doc, conjs[-1])[-1] + 1)

    sent_doc_ = [s.orth_ for s in sent_doc]

    sents_true = [(sent_doc_[:conj_poss[0]] + [conjs_.orth_] + sent_doc_[conj_poss[1]+1:])
                  for conjs_ in conjs]
    sents_false = copy.deepcopy(sents_true)
    repl_words = gen_wrong_word(conjs, num=num, rng=rng)
    if repl_words:
        sents_false = [(sent_doc_[:conj_poss[0]] + nltk.word_tokenize(rw) + sent_doc_[conj_poss[1]+1:])
                       for rw in repl_words]
    else:
        sents_false = change_neg(sents_true, root_pos)
    return sents_true, sents_false


def change_key(sent_doc, subtrees_, sent_key, num=5, rng=np.random.RandomState(156)):
#    sents_true = [s.orth_ for s in sent_doc]
    sents_false = []
    sent_doc_ = [s.orth_ for s in sent_doc]
    sents_true = sent_doc_
    change_funs = {'date': change_date, 'gpe': change_country}
    for key in sent_key.keys():
        try:
            if key not in change_funs:
                continue
            words_doc = [subtrees_[sent_key[key][i][0]][sent_key[key][i][1]]
                         for i in range(len(sent_key[key]))]
            word_poss = [get_sent_pos(sent_doc, wd) for wd in words_doc]
            word_poss = sorted(word_poss, key=lambda x: x[1])
            w_orth = [w.orth_.lower() for w in words_doc]
            repl_words = change_funs[key](w_orth, num=num, rng=rng)
            sents_false_ = [copy.deepcopy(sent_doc_) for _ in range(len(repl_words))]
    
            for si, sf in enumerate(sents_false_):
                for rw, rp in zip(repl_words[si][::-1], word_poss[::-1]):
                    if rw:
                        del sf[rp[0]:rp[1]+1]
                        sf = sf[:rp[0]] + nltk.word_tokenize(rw) + sf[rp[0]:]
                        sents_false_[si] = sf
            sents_false += sents_false_
        except (TypeError, ValueError):
            continue
    return [sents_true], sents_false


def gen_wrong_word(words_doc, num=5, rng=np.random.RandomState(156)):
    countrie_names = get_country_names()
    w_orth = [w.orth_.lower() for w in words_doc]
#    repl_words = []
    if np.any([is_country(w, countrie_names) for w in w_orth]):
        return change_country(w_orth, num=5, rng=rng)[:, 0]
    return []


def change_num(sent_doc):
    pass


def get_keys(labels_):
    keys = ['date', 'gpe', 'cardinal']
    keys_ = {}
    for lbi, lb in enumerate(labels_):
        for wi, w in enumerate(lb):
            try:
                if w.lower() in keys:
                    if w.lower() in keys_:
                        keys_[w.lower()].append((lbi, wi))
                    else:
                        keys_[w.lower()] = [(lbi, wi)]
            except AttributeError:
                continue
    return keys_


def change_country(w_orth, num=5, rng=np.random.RandomState(156)):
    """
    changes the country
    """
    countrie_names = get_country_names()
    country_last = np.array(list(set(countrie_names) - set(w_orth)))
    return np.reshape(country_last[rng.permutation(len(country_last))][:num*len(w_orth)], (num, -1))


def get_date_stat(dates):
    now = datetime.now()
    has_day = ['' if not dt else
               'd' if dt.day != now.day else '' for dt in dates]
    has_month = ['' if not dt else
                 'm' if dt.month != now.month else '' for dt in dates]
    has_year = ['' if not dt else
                'y' if dt.year != now.year else '' for dt in dates]

    date_stat = [d + m + y for d, m, y in zip(has_day, has_month, has_year)]
    date_stat = [ds if ds != '' else None for ds in date_stat]
    dsf = {'y': '%Y', 'm': '%B', 'd': '%d'}
    date_format = [None if not ds else ' '.join([dsf[ds[i]] for i in range(len(ds))])
                   for ds in date_stat]
    return date_stat, date_format


def strftime_(date, date_format):
    write_year = False
    if '%Y' in date_format:
        write_year = True
        date_format = date_format.replace('%Y', '').strip()
    year = date.year
    if year < 1900:
        date = date.replace(year=1900)
#    print date
    result = datetime.strftime(date, date_format).lower()
    if write_year:
        result += ' ' + str(year)
    date = date.replace(year=year)
    return result


def change_date(w_orth, num=5, rng=np.random.RandomState(156)):
    """
    changes the dates
    """
    dates = []
    for w in w_orth:
        try:
            dates += [date_parse(w)]
        except (ValueError, OverflowError):
            dates += [None]
    date_stat, date_format = get_date_stat(dates)
#    print len(dates), len(date_stat)
    
    years_ = rng.randint(-10, 10, size=num)
    months_ = rng.permutation(11)[:num]
    days_ = rng.permutation(26)[:num]
    days_delta = rng.randint(-100, 100, size=num)
#    dates_wrong = [[None if not dt else
#                    datetime(dt.year + years_[i], dt.month, dt.day) if ds[-1]=='y' else
#                    datetime(dt.year, list(set(range(1, 13))-set([dt.month]))[months_[i]], dt.day) if ds[-1]=='m' else
#                    datetime(dt.year, dt.month, list(set(range(1, 27))-set([dt.day]))[days_[i]])
#                    for dt, ds in zip(dates, date_stat)] for i in range(num)]
    dates_wrong = [[None if not dt else
                    dt.replace(year=dt.year + years_[i]) if ds[-1]=='y' else
                    dt + timedelta(days=days_delta[i]-1)
                    for dt, ds in zip(dates, date_stat)] for i in range(num)]
    dates_ = [[None if not d else strftime_(d, df).lower()
               for d, df in zip(dates_wrong_, date_format)] for dates_wrong_ in dates_wrong]
    return dates_


def get_antonym(word, art=['a']):
    antonyms = set()
    try:
        synsets_ = [ws for ws in wn.synsets(word) if ws.pos() in art]
        for s in synsets_:
            lemmas_ = s.lemmas()
            for l in lemmas_:
                try:
                    if l.name() in ['be', 'have', 'can', 'may', 'would']:
                        return []
                except ValueError:
                    return []
    #            print l.name()
                ant = l.antonyms()
                if ant:
                    antonyms.update([a.name() for a in ant])
    except (IndexError, StopIteration, ValueError, AssertionError):
        return []

    return list(antonyms)


def change_antonym(sent_doc, num=5, rng=np.random.RandomState(156)):
    """
    changes the sentence by changing words with correspondong antonyms
    from wordnet
    """
    sents_false = []
    sent_doc_ = [s.orth_ for s in sent_doc]
    sents_true = sent_doc_
    sents_false = []

    antonyms = {}

    for ws_ in sent_doc:
#        antonyms.append({})
#        for ws_ in st:
            if ws_.__class__.__name__ == 'Span':
                ws = [w for w in ws_]
            else:
                ws = [ws_]
            for w in ws:
                try:
                    art = getattr(wn, w.pos_)
                    ant = get_antonym(w.orth_, art=[art])
                    if ant:
                        antonyms[w] = get_antonym(w.orth_, art=[art])
                except AttributeError:
                    continue
    change_num = min(num, len(antonyms))
    word_poss = [(wd, get_sent_pos(sent_doc, wd)) for wd in antonyms.keys()]
    word_poss = sorted(word_poss, key=lambda x: x[1])

    sents_false_ = [copy.deepcopy(sent_doc_) for _ in range(change_num)]
    ant_perm = rng.permutation(len(antonyms))[:change_num]
    for si, sf in enumerate(sents_false_):
        rw, rp = word_poss[ant_perm[si]]
        if rw:
            del sf[rp[0]:rp[1]+1]
            sf = sf[:rp[0]] + nltk.word_tokenize(antonyms[rw][0]) + sf[rp[0]:]
            sents_false_[si] = sf
#            sf[rp[0]] = antonyms[rw][0]
    sents_false += sents_false_
    return [sents_true], sents_false


def change_name(sent_doc):
    pass


def change_adj(sent_doc):
    pass


def change_noun(sent_doc):
    pass


def change_neg(sent, not_pos):
    sent_ = copy.deepcopy(sent)
#    [sf.insert(root_pos, u'not') for sf in sents_]
    return [sf[:not_pos] + [u'not'] + sf[not_pos:] for sf in sent_]


def write_sents(sents_true, sents_false, file_from, save_file):
    """
    writes the generated sentences to the file
    """
    write_text = file_from + '\n'
    for s in sents_true:
        write_text += ' '.join(s) + '\t' + '1' + '\n'
    for s in sents_false:
        write_text += ' '.join(s) + '\t' + '0' + '\n'
    with open(save_file, 'w') as f:
        f.write(write_text.encode('utf-8'))


#def generate_sent_file(art_file, pos_count=1, neg_count=3,
#                  rng=np.random.RandomState(156), save_dir=None):
#    


def generate_sent(art_file, pos_count=1, neg_count=3,
                  rng=np.random.RandomState(156), save_dir=None, sf_prefix=1):
    with open(art_file, 'r') as f:
        art_text = f.read()

    sel_sents = select_sents(art_text, source.nlp, rng=rng)
    if sel_sents:
#        try:
        sents_true, sents_false = get_sent_mod(sel_sents, source.nlp, num=5, rng=rng)
#        except:
#            return None, None
#        pos_count = min(pos_count, len(sents_true))
        if len(sents_false) < neg_count or len(sents_true) < pos_count:
            return None, None
#        neg_count = min(neg_count, len(sents_false))
        sents_true_nums = rng.permutation(len(sents_true))[:pos_count]
        sents_false_nums = rng.permutation(len(sents_false))[:neg_count]
        sents_true = np.array(sents_true)[sents_true_nums]
        sents_false = np.array(sents_false)[sents_false_nums]
    
        if save_dir:
            save_file_name = '.'.join(os.path.basename(art_file).split('.')[:-1]) + \
                             '_' + str(sf_prefix) + '.txt'
            write_sents(sents_true, sents_false, art_file, save_dir + save_file_name)
        return sents_true, sents_false
    else:
        return None, None


def generate_sents_par(art_dir, n_jobs=8, sent_count=None, pos_count=1, neg_count=3,
                       rng=np.random.RandomState(156), save_dir=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    art_files = glob.glob(art_dir + '*.txt')
    (sents_true, sents_false) = zip(*Parallel(n_jobs=n_jobs, verbose=2)
                                    (delayed(generate_sent)(art_file, pos_count=pos_count, neg_count=neg_count,
                                                            rng=np.random.RandomState(156), save_dir=save_dir)
                                                            for art_file in art_files))
    return sents_true, sents_false


def generate_sents(art_dir, sent_count=None, pos_count=1, neg_count=3,
                   rng=np.random.RandomState(156), save_dir=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    art_files = glob.glob(art_dir + '*.txt')
    sents_true = []
    sents_false = []
    for ii, art_file in enumerate(art_files):
        sents_true_, sents_false_ = generate_sent(art_file, pos_count=pos_count, neg_count=neg_count,
                                                  rng=np.random.RandomState(156), save_dir=save_dir)
        sents_true.extend([sents_true_])
        sents_false.extend([sents_false_])
        _helpScripts.print_perc(float(ii) / float(len(art_files)) * 100 + 1,
                                suffix='performed {} from {}'.format(ii, len(art_files)))
    return sents_true, sents_false


def get_sents_cont(sents, wh_ix_dir=WHOOSH_DIR):
    wh_ix = whoosh.index.open_dir(wh_ix_dir)
    pass
#    sents_ =


def get_sent_cont(sents, wh_ix, except_files=[], res_num=10):
    """
    predicts the answer regarding the search score in the index
    between question and answer and the database
    """
#    print question
#    quest_num_train = len(questions)
    pred_answ = ''
#    print quest_num_train, 'questions to predict'
    qp = _wh_qparser.QueryParser("content", schema=wh_ix.schema)#,
#                                 group=whoosh.qparser.OrGroup)
    with wh_ix.searcher(weighting=_wh_scoring.BM25F()) as searcher:
        scores = np.zeros((4, res_num), dtype='float32')
        paths = np.zeros(((4, res_num)), dtype=object)
        for i in range(4):
#            phrase = question + ' ' + answers[i]
#            try:
#                phrase = unicode(phrase, 'utf-8')
#            except TypeError:
#                phrase
            if question:
                phrase = combine_quest_answ(question, answers[i])
            else:
                if isinstance(answers[i], list):
                    phrase = u' OR '.join(answers[i])
            q = qp.parse(phrase)
#            print question
#            print ':::', answers[i]
#            print '!!!!!!!!!!!!!!!\r\n'
            results = searcher.search(q, limit=res_num, terms=False)
#                for ir, res in enumerate(results):
#                    print 'Score: ', results[ir].score
#                    print results[ir]
#                    print '!!!!!!!!!!!!!!!!'
            try:
#                scores[i] = results[0].score
                scores[i, :] = np.array([res.score for res in results]).reshape((1, -1))
                paths[i, :] = np.array([res["path"] for res in results]).reshape((1, -1))
            except (IndexError, ValueError):
                paths = ['no file']
                continue
#                except KeyError:
#                    continue
#            scores = Parallel(n_jobs=4)(delayed(search_phrase_score)(searcher, qp,
#                                                                     questions[r], answers[r],i)
#                                        for i in range(4))
        pred_answ = answer_sym[np.argmax(np.max(scores, axis=1))]
#        print pred_answ
#    finally:
#        searcher.close()
    return pred_answ, scores, paths


def get_full_sents(sents_true, sents_false):
    leave_idx = []
    for i in range(len(sents_true))[::-1]:
        if sents_true[i] == None:
            continue
        else:
            leave_idx += [i]
    return leave_idx


if __name__ == '__main__':
    os.chdir('/Users/lexx/Documents/Work/Kaggle/AllenAIScience/')

    save_dir = './data/generated_sents/'
    source = _aai_data.Source()

    art_dirs_pre = u'./data/corpus/'
    wiki_dir = u'wiki_text_mod_spl_50/'
    ck12_full_spl_dir = u'ck12_full_themes_mod_spl_50/'
    ck12_wiki_spl_dir = u'wiki_text_ck12_themes_mod_spl_50/'
    art_dirs = [wiki_dir, ck12_full_spl_dir, ck12_wiki_spl_dir]

    _helpScripts.print_msg('Generate sentences')
    (sents_true, sents_false) = generate_sents_par(art_dirs_pre+art_dirs[2],
                                               pos_count=1, neg_count=3,
                                               rng=np.random.RandomState(156),
                                               save_dir=save_dir+art_dirs[2])
    sents_true = np.array(sents_true)
    sents_false = np.array(sents_false)
    full_idx = get_full_sents(sents_true, sents_false)
    sents_true = sents_true[full_idx]
    sents_false = sents_false[full_idx]
    


def temp():
    art_files = [glob.glob(art_dirs_pre + art_dir + '*.txt') for art_dir in art_dirs]
    
    with open(art_files[1][134], 'r') as f:
        art_text = f.read()

    art_sents = nltk.sent_tokenize(art_text)
    sent_ = art_sents[7]
    sent_doc = source.nlp(unicode(sent_))
    _questRewr.is_sentence(sent_doc)
    
    
    
    
    good = wn.synset('good.a.01')
    good.antonyms()