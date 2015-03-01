'''
PyBrain MNIST Example / Tutorial

- Pulled from 
    http://sujitpal.blogspot.com/2014/07/handwritten-digit-recognition-with.html

'''
# Source: src/digit_recognition/neural_network.py
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pylab import imshow, show, cm

def load_dataset(dataset, X, y):
    enc = OneHotEncoder(n_values=10)
    yenc = enc.fit_transform(np.matrix(y)).todense()
    for i in range(y.shape[0]):
        dataset.addSample(X[i, :], yenc[i][0])

NUM_EPOCHS = 50
NUM_HIDDEN_UNITS = 25

print "Loading MATLAB data..."    
data = scipy.io.loadmat("../../data/digit_recognition/ex3data1.mat")
X = data["X"]
y = data["y"]
y[y == 10] = 0 # '0' is encoded as '10' in data, fix it
n_features = X.shape[1]
n_classes = len(np.unique(y))

# visualize data
# get 100 rows of the input at random
print "Visualize data..."
idxs = np.random.randint(X.shape[0], size=100)
fig, ax = plt.subplots(10, 10)
img_size = math.sqrt(n_features)
for i in range(10):
    for j in range(10):
        Xi = X[idxs[i * 10 + j], :].reshape(img_size, img_size).T
        ax[i, j].set_axis_off()
        ax[i, j].imshow(Xi, aspect="auto", cmap="gray")
plt.show()

# split up training data for cross validation
print "Split data into training and test sets..."
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, 
                                                random_state=42)
ds_train = ClassificationDataSet(X.shape[1], 10)
load_dataset(ds_train, Xtrain, ytrain)

# build a 400 x 25 x 10 Neural Network
print "Building %d x %d x %d neural network..." % (n_features, 
                                                   NUM_HIDDEN_UNITS, n_classes)
fnn = buildNetwork(n_features, NUM_HIDDEN_UNITS, n_classes, bias=True, 
                   outclass=SoftmaxLayer)
print fnn

# train network
print "Training network..."
trainer = BackpropTrainer(fnn, ds_train)
for i in range(NUM_EPOCHS):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error)
    
# predict using test data
print "Making predictions..."
ypreds = []
ytrues = []
for i in range(Xtest.shape[0]):
    pred = fnn.activate(Xtest[i, :])
    ypreds.append(pred.argmax())
    ytrues.append(ytest[i])
print "Accuracy on test set: %7.4f" % accuracy_score(ytrues, ypreds, 
                                                     normalize=True)
