import logging
import sys
import numpy
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from random import shuffle
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, \
    accuracy_score, roc_curve, roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("running %s" % ' '.join(sys.argv))


class LabeledLineSentence(object):

    def __init__(self, docs, labels, sources=None):
        self.sources = sources
        self.docs = docs
        self.labels = labels
        self.sentences = []

        flipped = {}

        if self.sources:
            # make sure that keys are unique
            for key, value in sources.items():
                if value not in flipped:
                    flipped[value] = [key]
                else:
                    raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        if self.sources:
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
        else:
            item_no = 0
            for doc, label in zip(self.docs, self.labels):
                item_no += 1
                yield TaggedDocument(utils.to_unicode(doc).split(), [str(label) + '_%s' % item_no])

    def to_array(self):
        if self.sources:
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        self.sentences.append(TaggedDocument(
                            utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        else:
            item_no = 0
            for doc, label in zip(self.docs, self.labels):
                self.sentences.append(TaggedDocument(
                    utils.to_unicode(doc).split(), [str(label) + '_%s' % item_no]))
                item_no += 1
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def model2vec(model):
    X = model.docvecs.doctag_syn0
    y = model.docvecs.doctags
    y = [label[0] for label in y]
    joblib.dump((X, y), 'data/docvecs.data')
    return X, y


def get_knn_results(X, y, n_neighbors=2, n_jobs=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    cls = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    cls.fit(X_train, y_train)
    results = cls.predict(X_test)

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
    plt.legend(loc="lower right")
    plt.show()
    return results



