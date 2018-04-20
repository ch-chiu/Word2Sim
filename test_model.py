import os
import json
import logging
import numpy as np
from time import time
from gensim.models import KeyedVectors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from word_movers_knn import WordMoversKNN, WordMoversKNNCV

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Initialize")
    t0 = time()

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

    # Load 20newsgroup data and split train/test set
    newsgroups = fetch_20newsgroups()
    docs, y = newsgroups.data, newsgroups.target
    docs_train, docs_test, y_train, y_test = train_test_split(docs, y,
                                                              train_size=100,
                                                              test_size=300,
                                                              random_state=0)
    # Transfer bag of words to sparse matrix
    vect = CountVectorizer(stop_words="english").fit(docs_train + docs_test)
    # Only maintain words occur in 20newsgroup to reduce volume of vectors and vocabulary dict
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    # Create fixed-vocabulary vectorizer using only the words we have embeddings for
    vect = CountVectorizer(vocabulary=common, dtype=np.double)
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    t1 = time()
    logger.info("Finished loading data in %s", t1 - t0)

    # Training WordMoversKNNCV model
    knn_cv = WordMoversKNNCV(cv=3,
                             n_neighbors_try=range(1, 5),
                             W_embed=W_common, verbose=5, n_jobs=8)
    logger.info("Starting training model.")
    knn_cv.fit(X_train, y_train)
    joblib.dump(knn_cv, "knncv_model.m")
    print("CV score: {:.2f}".format(knn_cv.cv_scores_.mean(axis=0).max()))