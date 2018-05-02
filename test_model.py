import os
import logging
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, \
    accuracy_score, roc_curve, roc_auc_score
from sklearn.externals import joblib
from sentiment import *
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Initialize")
    t0 = time()

    # Read vectors and vocabulary from disk
    W = np.memmap("data/embed.dat", dtype=np.float64, mode="r", shape=(3000000, 300))
    with open("data/embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())

    # Build vocabulary dictionary
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}

    # Load IMDB data and split train/test set
    train_pos, train_neg, test_pos, test_neg = load_data('data/imdb/')
    docs_train = train_pos + train_neg
    docs_test = test_pos + test_neg
    y_train = np.c_[np.ones((1, len(train_pos)), dtype=np.float64),
                    np.zeros((1, len(train_neg)), dtype=np.float64)][0]
    y_test = np.c_[np.ones((1, len(test_pos)), dtype=np.float64),
                   np.zeros((1, len(test_neg)), dtype=np.float64)][0]

    # Transfer bag of words to sparse matrix
    vect = TfidfVectorizer(stop_words="english").fit(docs_train + docs_test)
    # Only maintain words occur in 20newsgroup to reduce volume of vectors and vocabulary dict
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    # Create fixed-vocabulary vectorizer using only the words we have embeddings for
    vect = TfidfVectorizer(vocabulary=common, dtype=np.double)
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    logger.info("Loading model.")
    model = joblib.load('model/imdb_WMS.m')
    results = model.predict(X_test)

    print("Classfication Report: %s\n", classification_report(y_test, results))
    print("Confusion Matrix: %s\n", confusion_matrix(y_test, results))
    print("Precision: {}".format(precision_score(y_test, results, average='weighted')))
    print("Recall: {}".format(recall_score(y_test, results, average='weighted')))
    print("Accuracy: {}".format(accuracy_score(y_test, results)))
    roc_auc = roc_auc_score(y_test, results)
    fpr, tpr, thresholds = roc_curve(y_test, results)
    plt.figure()
    plt.plot(fpr, tpr, label='K-NN ROC curve (area = %0.6f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-NN Classification ROC AUC (WMS)')
    # plt.legend(loc="lower right")
    plt.show()