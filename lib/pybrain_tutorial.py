'''
PyBrain MNIST Example / Tutorial

- Pulled from 
    http://martin-thoma.com/classify-mnist-with-pybrain/
    http://sujitpal.blogspot.com/2014/07/handwritten-digit-recognition-with.html

'''
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pylab import imshow, show, cm


def load_data(filename):
    pass


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

def split_data(X, y):
    # split up training data for cross validation
    print "Split data into training and test sets..."
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, 
                                                    random_state=42)


def classify(training, testing, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, num_epochs):
    classes = 10
    num_features = testing['rows'] * testing['cols']


    
    ds_train = ClassificationDataSet(X.shape[1], 10)
    load_dataset(ds_train, Xtrain, ytrain)

    # build a 400 x 25 x 10 Neural Network
    print "Building %d x %d x %d neural network..." % (num_features,num_hidden_units, n_classes)

    fnn = buildNetwork(num_features, num_hidden_units, n_classes, bias=True, outclass=SoftmaxLayer)
    
    print fnn

    # train network
    print "Training network..."
    trainer = BackpropTrainer(fnn, ds_train)
    
    for i in range(num_epochs):
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



if __name__ == '__main__':
    num_epochs = 50
    num_hidden_units = 25


    classify(training, testing, hidden_neurons, momentum,
             weightdecay, learning_rate, lrdecay, num_epochs)