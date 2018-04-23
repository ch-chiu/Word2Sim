import os
import sys
import logging
import numpy as np
from time import time
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, \
    accuracy_score, roc_curve, roc_auc_score
from sklearn.externals import joblib
from word_movers_knn import WordMoversKNN, WordMoversKNNCV
from sentiment import *
import matplotlib.pyplot as plt

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
        newsgroups = fetch_20newsgroups(categories=catrgories)
        docs, y = newsgroups.data, newsgroups.target
        docs_train, docs_test, y_train, y_test = train_test_split(docs, y,
                                                                  train_size=400,
                                                                  test_size=100,
                                                                  random_state=0)
    elif data_set == 'imdb':
        # Load IMDB data and split train/test set
        train_pos, train_neg, test_pos, test_neg = load_data('data/imdb/')
        docs_train = train_pos + train_neg
        docs_test = test_pos + test_neg
        y_train = np.c_[np.ones((1, len(train_pos)), dtype=np.float64),
                        np.zeros((1, len(train_neg)), dtype=np.float64)][0]
        y_test = np.c_[np.ones((1, len(test_pos)), dtype=np.float64),
                       np.zeros((1, len(test_neg)), dtype=np.float64)][0]
    elif data_set == 'twitter':
        # Load twitter data and split train/test set
        train_pos, train_neg, test_pos, test_neg = load_data('data/twitter/')
        docs_train = train_pos + train_neg
        docs_test = test_pos + test_neg
        y_train = np.c_[np.ones((1, len(train_pos)), dtype=np.float64),
                        np.zeros((1, len(train_neg)), dtype=np.float64)][0]
        y_test = np.c_[np.ones((1, len(test_pos)), dtype=np.float64),
                       np.zeros((1, len(test_neg)), dtype=np.float64)][0]
    else:
        raise SystemExit('No such dataset.')

    # Transfer bag of words to sparse matrix
    vect = TfidfVectorizer(stop_words="english").fit(docs_train + docs_test)
    # Only maintain words occur in 20newsgroup to reduce volume of vectors and vocabulary dict
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    # Create fixed-vocabulary vectorizer using only the words we have embeddings for
    vect = TfidfVectorizer(vocabulary=common, dtype=np.double)
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    t1 = time()
    logger.info("Finished loading data in %fs", t1 - t0)

    if model_type == 'WMS_CV':
        # Training WordMoversKNNCV model
        model = WordMoversKNNCV(cv=3,
                                n_neighbors_try=range(1, 5),
                                W_embed=W_common, verbose=5, n_jobs=8)
        logger.info("Starting training model.")
        model.fit(X_train, y_train)
        joblib.dump(model, "model/" + data_set + '_' + model_type + '.m')
        print("CV score: {:.2f}".format(model.cv_scores_.mean(axis=0).max()))
    elif model_type == 'WMS':
        # Training WordMoversKNN model
        model = WordMoversKNN(W_embed=W_common, verbose=5, n_jobs=8)
        logger.info("Starting training model.")
        model.fit(X_train, y_train)
        joblib.dump(model, "model/" + data_set + '_' + model_type + '.m')
    # elif model_type == 'TF-IDF':

    results = model.predict(X_test)

    print("Classfication Report: %s\n", classification_report(y_test, results))
    print("Confusion Matrix: %s\n", confusion_matrix(y_test, results))
    print("Precision: {}".format(precision_score(y_test, results, average='weighted')))
    print("Recall: {}".format(recall_score(y_test, results, average='weighted')))
    print("Accuracy: {}".format(accuracy_score(y_test, results)))
    roc_auc = roc_auc_score(y_test, results)
    fpr, tpr = roc_curve(y_test, results)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression ROC curve (area = %0.6f)' % roc_auc)

