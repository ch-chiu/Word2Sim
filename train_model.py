import os
import sys
import json
import logging
import numpy as np
from time import time
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib
from gensim.models import Doc2Vec
from gensim.similarities import WmdSimilarity
from sklearn.neighbors import KNeighborsClassifier
from word_movers_knn import WordMoversKNN, WordMoversKNNCV
from utilities import *
from mydoc2vec import LabeledLineSentence, model2vec, get_knn_results
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, \
    accuracy_score, roc_curve, roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

catrgories = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x"
]


if __name__ == '__main__':
    logger.info("Initialize")
    t0 = time()

    if len(sys.argv) != 3:
        print(len(sys.argv))
        raise SystemExit("Usage: Python train_model.py <data_set> <model_type>")
    data_set, model_type = sys.argv[1:3]
    model_dir = "model/" + data_set + '_' + model_type + '.m'

    # Write vectors and vocabulary form GoogleNews-vectors into disk
    if not os.path.exists("data/embed.dat"):
        print("Caching word embeddings in memmapped format...")

        wv = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        fp = np.memmap("data/embed.dat", dtype=np.float64, mode='w+', shape=wv.syn0.shape)
        fp[:] = wv.syn0[:]
        with open("data/embed.vocab", "w") as f:
            for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                print(w, file=f)
        del fp, wv

    # Read vectors and vocabulary from disk
    W = np.memmap("data/embed.dat", dtype=np.float64, mode="r", shape=(3000000, 300))
    with open("data/embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())

    # Build vocabulary dictionary
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}

    if data_set == '20newsgroup':
        # Load 20newsgroup data and split train/test set
        class_len = len(catrgories)
        newsgroups = fetch_20newsgroups(categories=catrgories)
        docs, y = newsgroups.data, newsgroups.target
        docs_train, docs_test, y_train, y_test = train_test_split(docs, y,
                                                                  test_size=0.25,
                                                                  random_state=0)
    elif data_set == 'imdb':
        # Load IMDB data and split train/test set
        class_len = 2
        train_pos, train_neg, test_pos, test_neg = load_data('data/imdb/')
        docs_train = train_pos + train_neg
        docs_test = test_pos + test_neg
        docs = docs_train + docs_test
        y_train = np.c_[np.ones((1, len(train_pos)), dtype=np.float64),
                        np.zeros((1, len(train_neg)), dtype=np.float64)][0]
        y_test = np.c_[np.ones((1, len(test_pos)), dtype=np.float64),
                       np.zeros((1, len(test_neg)), dtype=np.float64)][0]
        y = np.r_[y_train, y_test]
    elif data_set == 'twitter':
        # Load twitter data and split train/test set
        class_len = 2
        train_pos, train_neg, test_pos, test_neg = load_data('data/twitter/')
        docs_train = train_pos + train_neg
        docs_test = test_pos + test_neg
        docs = docs_train + docs_test
        y_train = np.c_[np.ones((1, len(train_pos)), dtype=np.float64),
                        np.zeros((1, len(train_neg)), dtype=np.float64)][0]
        y_test = np.c_[np.ones((1, len(test_pos)), dtype=np.float64),
                       np.zeros((1, len(test_neg)), dtype=np.float64)][0]
        y = np.r_[y_train, y_test]
    elif data_set == 'esx':
        # Load esx bug data and split train/test set
        class_len = 50
        with open('data/esx/trainData.json') as f:
            train_data = json.load(f)
        with open('data/esx/testData.json') as f:
            test_data = json.load(f)
        docs_train = [doc['short_desc'] for doc in train_data]
        docs_test = [doc['short_desc'] for doc in test_data]
        y_train = [doc['component_id'] for doc in train_data]
        y_test = [doc['component_id'] for doc in test_data]
        docs = docs_train + docs_test
        y = np.r_[y_train, y_test]
    else:
        raise SystemExit('No such dataset.')

    joblib.dump(docs_train, 'data/docs_train.dat')
    joblib.dump(docs_test, 'data/docs_test.dat')
    joblib.dump(y_train, 'data/y_train.dat')
    joblib.dump(y_test, 'data/y_test.dat')

    # Transfer bag of words to sparse matrix
    vect = TfidfVectorizer(stop_words="english").fit(docs_train + docs_test)
    count_vect = CountVectorizer(stop_words="english").fit(docs_train + docs_test)
    # Only maintain words occur in 20newsgroup to reduce volume of vectors and vocabulary dict
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    # Create fixed-vocabulary vectorizer using only the words we have embeddings for
    vect = TfidfVectorizer(vocabulary=common, dtype=np.double)
    count_vect = CountVectorizer(vocabulary=common, dtype=np.double)
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    joblib.dump(X_train, 'data/X_train.dat')
    joblib.dump(X_test, 'data/X_test.dat')

    logger.info("Finished loading data in %fs", time() - t0)

    if model_type == 'WMS_CV':
        # Training WordMoversKNNCV model
        model = WordMoversKNNCV(cv=3,
                                n_neighbors=class_len,
                                W_embed=W_common, verbose=5, n_jobs=8)
        logger.info("Starting training model.")
        model.fit(X_train, y_train)
        results = model.predict(X_test)
        joblib.dump(model, model_dir)
        print("CV score: {:.2f}".format(model.cv_scores_.mean(axis=0).max()))
    elif model_type == 'WMS':
        # Training WordMoversKNN model
        model = WordMoversKNN(W_embed=W_common,
                              verbose=5, n_jobs=8)
        logger.info("Starting training model.")
        model.fit(X_train, y_train)
        results = model.predict(X_test)
        joblib.dump(model, model_dir)
    elif model_type == 'doc2vec':
        # # Training Doc2Vec model
        # sentences = LabeledLineSentence(docs, y).to_array()
        # model = Doc2Vec(min_count=2, window=5,
        #                 size=100, workers=8)
        # model.build_vocab(sentences)
        # for epoch in range(50):
        #     logger.info('Epoch %d' % epoch)
        #     model.train(sentences,
        #                 total_examples=model.corpus_count,
        #                 epochs=epoch)
        # model.save(model_dir)
        # model2vec(model)
        # For debugging, uncomment this line
        model = Doc2Vec.load('model/esx_doc2vec.m')
        array, label = model2vec(model)
        results = get_knn_results(array, label, n_neighbors=class_len, n_jobs=8)
        joblib.dump(model, model_dir)
    elif model_type == 'average':
        doc_vect, doc_tags = get_avg_vector(docs, y, count_vect, wv=W_common)
        joblib.dump((doc_vect, doc_tags), model_dir)
        results = get_knn_results(doc_vect, doc_tags, n_neighbors=class_len, n_jobs=8)
    else:
        raise SystemExit("Please indicate correct model type.")







