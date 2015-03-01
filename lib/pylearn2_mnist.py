'''
PyLearn2 MNIST Example / Tutorial

'''


import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class LogisticRegression(Model):
    def __init__(self, nvis, nclasses):
        super(LogisticRegression, self).__init__()

        self.nvis = nvis
        self.nclasses = nclasses

        # Randomly samples weights from uniform distribution for the structure size
        W_value = numpy.random.uniform(size=(self.nvis, self.nclasses))
        # Shared prepares GPU compatible code 
        self.W = sharedX(W_value, 'W')
        # Generates bias per ouptput 
        b_value = numpy.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')
        self._params = [self.W, self.b]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    def logistic_regression(self, inputs):
        # Equation run on each hidden node
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)


class LogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    # Cost will expect inputs and targets
    supervised = True 

    def expr(self, model, data, **kwargs):
        # Input checking
        space, source = self.get_data_specs(model)
        # Confirms data received is what was requested
        space.validate(data) 
        
        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()

