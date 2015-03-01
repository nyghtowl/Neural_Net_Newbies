'''
Scikit-Learn RBM MNIST Example / Tutorial

Code pulled directly from: http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/
'''
# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import time
import cv2

def load_digits(datasetPath):
    # build the dataset and then split it into data
    # and labels
    X = np.genfromtxt(datasetPath, delimiter = ",", dtype = "uint8")
    y = X[:, 0]
    X = X[:, 1:]

    # return a tuple of the data and targets
    return (X, y)

def scale(X, eps = 0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

def nudge(X, y):
    # initialize the translations to shift the image one pixel
    # up, down, left, and right, then initialize the new data
    # matrix and targets
    translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    data = []
    target = []

    # loop over each of the digits
    for (image, label) in zip(X, y):
        # reshape the image from a feature vector of 784 raw
        # pixel intensities to a 28x28 'image'
        image = image.reshape(28, 28)

        # loop over the translations
        for (tX, tY) in translations:
            # translate the image
            M = np.float32([[1, 0, tX], [0, 1, tY]])
            trans = cv2.warpAffine(image, M, (28, 28))

            # update the list of data and target
            data.append(trans.flatten())
            target.append(label)

    # return a tuple of the data matrix and targets
    return (np.array(data), np.array(target))

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True,
        help = "path to the dataset file")
    ap.add_argument("-t", "--test", required = True, type = float,
        help = "size of test split")
    ap.add_argument("-s", "--search", type = int, default = 0,
        help = "whether or not a grid search should be performed")
    args = vars(ap.parse_args())

    # load the digits dataset, convert the data points from integers
    # to floats, and then scale the data s.t. the predictors (columns)
    # are within the range [0, 1] -- this is a requirement of the
    # Bernoulli RBM
    (X, y) = load_digits(args["dataset"])
    X = X.astype("float32")
    X = scale(X)

    # construct the training/testing split
    (trainX, testX, trainY, testY) = train_test_split(X, y,
        test_size = args["test"], random_state = 42)

    # check to see if a grid search should be done
    if args["search"] == 1:
        # perform a grid search on the 'C' parameter of Logistic
        # Regression
        print "SEARCHING LOGISTIC REGRESSION"
        params = {"C": [1.0, 10.0, 100.0]}
        start = time.time()
        gs = GridSearchCV(LogisticRegression(), params, n_jobs = -1, verbose = 1)
        gs.fit(trainX, trainY)

        # print diagnostic information to the user and grab the
        # best model
        print "done in %0.3fs" % (time.time() - start)
        print "best score: %0.3f" % (gs.best_score_)
        print "LOGISTIC REGRESSION PARAMETERS"
        bestParams = gs.best_estimator_.get_params()

        # loop over the parameters and print each of them out
        # so they can be manually set
        for p in sorted(params.keys()):
            print "\t %s: %f" % (p, bestParams[p])

        # initialize the RBM + Logistic Regression pipeline
        rbm = BernoulliRBM()
        logistic = LogisticRegression()
        classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

        # perform a grid search on the learning rate, number of
        # iterations, and number of components on the RBM and
        # C for Logistic Regression
        print "SEARCHING RBM + LOGISTIC REGRESSION"
        params = {
            "rbm__learning_rate": [0.1, 0.01, 0.001],
            "rbm__n_iter": [20, 40, 80],
            "rbm__n_components": [50, 100, 200],
            "logistic__C": [1.0, 10.0, 100.0]}

        # perform a grid search over the parameter
        start = time.time()
        gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
        gs.fit(trainX, trainY)

        # print diagnostic information to the user and grab the
        # best model
        print "\ndone in %0.3fs" % (time.time() - start)
        print "best score: %0.3f" % (gs.best_score_)
        print "RBM + LOGISTIC REGRESSION PARAMETERS"
        bestParams = gs.best_estimator_.get_params()

        # loop over the parameters and print each of them out
        # so they can be manually set
        for p in sorted(params.keys()):
            print "\t %s: %f" % (p, bestParams[p])

        # show a reminder message
        print "\nIMPORTANT"
        print "Now that your parameters have been searched, manually set"
        print "them and re-run this script with --search 0"

    # otherwise, use the manually specified parameters
    else:
        # evaluate using Logistic Regression and only the raw pixel
        # features (these parameters were cross-validated)
        logistic = LogisticRegression(C = 1.0)
        logistic.fit(trainX, trainY)
        print "LOGISTIC REGRESSION ON ORIGINAL DATASET"
        print classification_report(testY, logistic.predict(testX))

        # initialize the RBM + Logistic Regression classifier with
        # the cross-validated parameters
        rbm = BernoulliRBM(n_components = 200, n_iter = 40,
            learning_rate = 0.01,  verbose = True)
        logistic = LogisticRegression(C = 1.0)

        # train the classifier and show an evaluation report
        classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
        classifier.fit(trainX, trainY)
        print "RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET"
        print classification_report(testY, classifier.predict(testX))

        # nudge the dataset and then re-evaluate
        print "RBM + LOGISTIC REGRESSION ON NUDGED DATASET"
        (testX, testY) = nudge(testX, testY)
        print classification_report(testY, classifier.predict(testX))

if __name__ == '__main__':
    main()