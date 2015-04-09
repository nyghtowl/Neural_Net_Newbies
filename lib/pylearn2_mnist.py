'''
PyLearn2 MNIST Example / Tutorial

'''

import os
from pylearn2.utils import serial
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.space import CompositeSpace
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from subprocess import check_call
import numpy as np

def load_data(dataset):
    if os.path.isfile(dataset):
        if not os.path.isfile("data/pylearn2/mnist_train_X.pkl"):
            split_data(dataset)
    else:
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)

def split_data(dataset):
    path = dataset.split("/")[0:2]
    check_call(["gunzip" + dataset])

    data = serial.load(dataset) 
    serial.save(path +'mnist_train_X.pkl', data[0][0]) 
    serial.save(path +'mnist_train_y.pkl', data[0][1].reshape((-1, 1))) 
    serial.save(path +'mnist_valid_X.pkl', data[1][0]) 
    serial.save(path +'mnist_valid_y.pkl', data[1][1].reshape((-1, 1))) 
    serial.save(path +'mnist_test_X.pkl', data[2][0]) 
    serial.save(path +'mnist_test_y.pkl', data[2][1].reshape((-1, 1)))

class LogisticRegression(Model):
    def __init__(self, nvis, nclasses):
        super(LogisticRegression, self).__init__()

        # Number of input nodes
        self.nvis = nvis
        # Number of output nodes
        self.nclasses = nclasses

        W_value = np.random.uniform(size=(self.nvis, self.nclasses))
        self.W = sharedX(W_value, 'W') # sharedX formats for GPUs
        
        b_value = np.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')

        self._params = [self.W, self.b]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    # Linear transformation followed by non-linear softmax transformation
    def logistic_regression(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)

    # Following two add comments on error rate during training
    def get_monitoring_data_specs(self):
        space = CompositeSpace([self.get_input_space(),
                                self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        print '... evaluate model' 
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.logistic_regression(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()

        return OrderedDict([('error', error)])

class LogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data):
        space, source = self.get_data_specs(model)
        # Checks data is valid - tensor variables expected
        space.validate(data)
        
        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        # Negative log likelihood
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()

def main(dataset='data/pylearn2/mnist.pkl.gz', nn_config = "lib/pylearn2_log.yaml"):
    print '... load and setup data'
    load_data(dataset)

    print '... building the model structure'
    train_obj = serial.load_train_file(nn_config)

    print '... training the model'
    train_obj.main_loop()



if __name__ == '__main__':
    main()