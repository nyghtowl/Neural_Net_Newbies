'''
Scikit-Learn RBM MNIST Example / Tutorial

'''
# import the necessary packages
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np

def load_data(datasetPath='data/'):
    return fetch_mldata('MNIST original', data_home=datasetPath)

def scale(X, eps = 0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

def split_data(data, test_size, random_state):
    X = data['data']
    y = data['target']
    X = X.astype("float32")
    X = scale(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main(test_size=0.1, n_iter = 10, learning_rate = 0.01):

    n_components = 200 # number of neurons in hidden layer
    verbose = True
    
    print '... load and setup data'
    dataset = load_data()
    trainX, testX, trainY, testY = split_data(dataset, test_size=test_size, random_state=42)

    print '... building the model structure'
    # initialize the RBM + Logistic Regression classifier with the cross-validated parameters

    rbm = BernoulliRBM(n_components = n_components, n_iter = n_iter,
        learning_rate = learning_rate)

    logistic = LogisticRegression(C = 1.0)

    print '... training the model'
    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)

    # rbm.fit(trainX, trainY)

    print '... evaluate model' 
    # Predict does not exist and its not clear how to get value out
    print classifier.score(testX, testY)



if __name__ == '__main__':
    main()