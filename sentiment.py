import numpy


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

    return train_pos[0:200], train_neg[0:200], test_pos[0:50], test_neg[0:50]


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    res = model.predict(test_pos_vec)
    unique, counts = numpy.unique(res, return_counts=True)
    d = dict(zip(unique, counts))
    tp = float(d['pos'])
    fn = float(d['neg'])
    
    res = model.predict(test_neg_vec)
    unique, counts = numpy.unique(res, return_counts=True)
    d = dict(zip(unique, counts))
    tn = float(d['neg'])
    fp = float(d['pos'])
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d" % (tp, fn))
        print("neg\t\t%d\t%d" % (fp, tn))
    print("accuracy: %f" % accuracy)


