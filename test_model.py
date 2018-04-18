import os
import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from word_movers_knn import WordMoversKNN, WordMoversKNNCV

W_common = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
wv = np.array(W_common.vectors, dtype=np.float64)

categories = [
    # 'alt.atheism',
    # 'comp.graphics',
    # 'comp.os.ms-windows.misc',
    # 'comp.sys.ibm.pc.hardware',
    # 'comp.sys.mac.hardware',
    # 'comp.windows.x',
    # 'misc.forsale',
    # 'rec.autos',
    # 'rec.motorcycles',
    # 'rec.sport.baseball',
    # 'rec.sport.hockey',
    # 'sci.crypt',
    # 'sci.electronics',
    # 'sci.med',
    # 'sci.space',
    # 'soc.religion.christian',
    # 'talk.politics.guns',
    # 'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
X_train = count_vect.fit_transform(twenty_train.data)

# tfidf_transformer = TfidfTransformer()
# X_train = tfidf_transformer.fit_transform(X_train)
y_train = twenty_train.target

# X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, twenty_train.target, test_size=.4, random_state=None)

knn_cv = WordMoversKNNCV(cv=3,
                         n_neighbors_try=range(1, 5),
                         W_embed=wv, verbose=5, n_jobs=3)
knn_cv.fit(X_train, y_train)
print("CV score: {:.2f}".format(knn_cv.cv_scores_.mean(axis=0).max()))