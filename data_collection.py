#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Zimeng Qiu <zqiu@vmware.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This script generate a tokenized json doc of corpus
"""

import os
import json
import logging
import sys

from xml.dom.minidom import parse
from preprocessor import Preprocessor
import xml.dom.minidom
import gensim

dir_path = os.path.dirname(os.path.realpath(__file__))


def write2json(docs, out_dir, json_file):
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as exception:
            raise SystemExit("Error: Could not create the output dir.")
    with open(json_file, 'w') as f:
        json.dump(docs, f, indent=3)


def load_trained_vector():
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
    print(model['love'])


def parse_xml():
    DOMTree = xml.dom.minidom.parse("truth_data_nyt_2017_v2.3.xml")
    collection = DOMTree.documentElement
    if collection.hasAttribute("shelf"):
        print("Root element : %s" % collection.getAttribute("shelf"))

    topics = collection.getElementsByTagName("topic")
    docs = list()
    text = list()

    for topic in topics:
        corpus = dict()
        if topic.hasAttribute("name"):
            corpus['name'] = topic.getAttribute("name")
            print("Topic Name: %s" % corpus['name'])
        if topic.hasAttribute("id"):
            corpus['id'] = topic.getAttribute("id")
            print("ID: %s" % corpus['id'])
        corpus['description'] = topic.getElementsByTagName('description')[0].childNodes[0].data
        print("Description: %s" % corpus['description'])
        corpus['narrative'] = topic.getElementsByTagName('narrative')[0].childNodes[0].data
        print("Narrative: %s" % corpus['narrative'])

        subtopics = topic.getElementsByTagName('subtopic')
        subtopic_list = list()
        for subtopic in subtopics:
            sub_corpus = dict()
            if subtopic.hasAttribute("name"):
                sub_corpus['name'] = subtopic.getAttribute("name")
                print("Topic Name: %s" % sub_corpus['name'])
            if subtopic.hasAttribute("id"):
                sub_corpus['id'] = subtopic.getAttribute("id")
                print("ID: %s" % sub_corpus['id'])

            passages = subtopic.getElementsByTagName('passage')
            passage_list = list()
            for passage in passages:
                pa_corpus = dict()
                if passage.hasAttribute('id'):
                    pa_corpus['id'] = passage.getAttribute("id")
                pa_corpus['text'] = passage.getElementsByTagName('text')[0].childNodes[0].data
                pa_corpus['type'] = passage.getElementsByTagName('type')[0].childNodes[0].data
                passage_list.append(pa_corpus)
                text.append(pa_corpus)

            sub_corpus['passages'] = passage_list
            subtopic_list.append(sub_corpus)
        corpus['subtopics'] = subtopic_list
        docs.append(corpus)

    return text


if __name__ == '__main__':
    document = parse_xml()
    out_dir = os.path.join(dir_path, 'data')
    out_file = os.path.join(out_dir, 'corpus.json')
    write2json(document, out_dir, out_file)

    doc_list = list()
    tokened_docs = Preprocessor(doc_list=document, tokens_only=False)
    for doc in tokened_docs:
        doc_list.append(doc)
    tokened_file = os.path.join(out_dir, 'tokened_doc.json')
    write2json(doc_list, out_dir, tokened_file)