#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Zimeng Qiu <zqiu@vmware.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Text corpora usually reside on disk, as text files in one format or another
In a common scenario, we need to build a tokenized list of words.

This class read the dumped json format bugzilla raw data, do some preprocess
"""


from __future__ import with_statement

import logging
import os
import json
import csv
import re

from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim import utils
from gensim.models.doc2vec import TaggedDocument


logger = logging.getLogger('gensim.corpora.textcorpus')

STOPWORDS_LIST = "stopwords_CornellUniv.csv"

lem = WordNetLemmatizer()
stem = PorterStemmer()
tokenizer = TweetTokenizer()
dir_path = os.path.dirname(os.path.realpath(__file__))


class Preprocessor(object):
    """
    Helper class to simplify the pipeline of getting tokenized words from bugzilla json dump
    on the disk
    """
    def __init__(self, doc_list=[], tokens_only=False):
        self.doc_list = doc_list
        # a python list of documents to train the model
        self.tokens_only = tokens_only

    def get_stopwords(self):
        csv_file = os.path.join(dir_path, STOPWORDS_LIST)
        stop_words = []
        if not os.path.exists(csv_file):
            print('Stopwords list:"%s" does not exist, ignored' %STOPWORDS_LIST)
        else:
            csv_data = csv.reader(open(csv_file, 'r', encoding='utf-8', errors='ignore'))
            stop_words.extend(list(row[0] for row in csv_data))
            return stop_words

    def split_tri_gram(self, string):
        str_list = []
        for index, char in enumerate(string):
            if index + 2 < len(string):
                str_list.append(string[index:index+3])
        return str_list

    def __iter__(self):
        stop_words = self.get_stopwords()
        text = ""
        for count, doc in enumerate(self.doc_list):
            # text_list = list()
            if 'id' in doc:
                doc_id = doc['id']
            else:
                doc_id = 0
            if 'type' in doc:
                doc_type = doc['type']
            else:
                doc_type = 'N/A'
            text = doc['text']

            # Lemmatize the comments for better match
            words = text.replace('~', ' ').replace('`', ' ').replace('!', ' ').replace('@', ' ').replace('#', ' ').\
                replace('$', ' ').replace('%', ' ').replace('^', ' ').replace('&', ' ').replace('*', ' ').\
                replace('(', ' ').replace(')', ' ').replace('-', ' ').replace('_', ' ').replace('+', ' ').\
                replace('=', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']', ' ').\
                replace('|', ' ').replace('\'', ' ').replace(':', ' ').replace(';', ' ').replace('"', ' ').\
                replace('\"', ' ').replace('<', ' ').replace('>', ' ').replace(',', ' ').replace('.', ' ').\
                replace('?', ' ').replace('/', ' ')
            # words = re.split(",|.|/|-|[|]|(|)|_|\+|=|'|\"|;|\?|!|~|<|>|@|#|$|%|^|&|\*", text)
            words = utils.simple_preprocess(words, min_len=2, max_len=20)
            words = [word.lower() for word in words if word.lower() not in stop_words]
            words = [lem.lemmatize(word) for word in words]

            # words = [lem.lemmatize(word, pos="v") for word in words]
            # words = [lem.lemmatize(word, pos="a") for word in words]
            # words = [lem.lemmatize(word, pos="n") for word in words]

            doc = dict()
            doc['label'] = doc_id
            doc['type'] = doc_type
            doc['text'] = words

            if self.tokens_only:
                yield words
            else:
                yield doc
