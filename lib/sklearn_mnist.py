'''
Scikit-Learn RBM MNIST Example / Tutorial

Note: Found issue with running predictions on rbm structure. This script does not run fully.

Utilized code from to help: http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/
'''
# import the necessary packages
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
import numpy as np

def load_data(datasetPath='data/'):
    return fetch_mldata('MNIST original', data_home=datasetPath)

def scale(X, eps = 0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

def split_data(data, test_size, random_state):
    X = mnist['data']
    y = mnist['target']
    X = X.astype("float32")
    X = scale(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

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

def main(test_size=0.1, n_iter = 10, learning_rate = 0.01):

    n_components = 200 # number of neurons in hidden layer
    verbose = True
    
    print '... load and setup data'
    dataset = load_data()
    trainX, testX, trainY, testY = split_data(dataset, test_size=test_size, random_state=42)

    print '... building the model'
    rbm = BernoulliRBM(n_components = n_components, n_iter = n_iter,
        learning_rate = learning_rate)

    print '... training the model'
    rbm.fit(trainX, trainY)

    print '... evaluate model' 
    # Predict does not exist and its not clear how to get value out
    print classification_report(testY, rbm.predict(testX))

    # To get variations on test data for evaluation
    testX, testY = nudge(testX, testY)
    print classification_report(testY, rbm.predict(testX))

if __name__ == '__main__':
    main()