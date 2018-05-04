import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            # words = [w.lower() for w in line.strip().split() if len(w)>=3]
            # train_pos.append(words)
            train_pos.append(line)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            # words = [w.lower() for w in line.strip().split() if len(w)>=3]
            # train_neg.append(words)
            train_neg.append(line)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            # words = [w.lower() for w in line.strip().split() if len(w)>=3]
            # test_pos.append(words)
            test_pos.append(line)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            # words = [w.lower() for w in line.strip().split() if len(w)>=3]
            # test_neg.append(words)
            test_neg.append(line)

    return train_pos[0:1000], train_neg[0:1000], test_pos[0:100], test_neg[0:100]


def get_avg_vector(docs, tags, bow, wv):
    doc_vect = []
    doc_tags = []
    for doc, tag in zip(docs, tags):
        doc = [doc]
        count_vect = bow.transform(doc)
        vect = wv[count_vect.indices]
        doc_vect.append(np.mean(vect, axis=0))
        doc_tags.append(tag)
    return doc_vect, doc_tags


