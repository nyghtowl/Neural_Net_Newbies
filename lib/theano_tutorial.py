'''
Theano Multilayer Perceptron Net MNIST Example / Tutorial

'''

import theano
import theano.tensor as T

class Layer(object):
    '''
    Basic one layer
    '''
    def __init__(self, W_init, b_init, activation):
        n_output, n_input = W_init.shape
        assert b_init.shape == (n_output,)

        # weights
        self.W = theano.shared(value=W_init.astype(theano.config.floatX), name='W', borrow=True)
        #bias
        self.b = theano.shared(value=b_init.reshape(-1,1).astype(theano.config.floatX), name='b', borrow=True, broadcastable=(False,True))
        # activation function
        self.activation = activation
        #coefficients
        self.params = [self.W, self.b]

    def output(self, x): 
        lin_output = T.dot(self.W, x) + self.b 
        return (lin_output if self.activation is None else self.activation(lin_output))
    

class MLP(object):
    '''
    Multilayer perceptron

    Contains main structure (layers and parameters) & functionality
    '''
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
        self.layers = []
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))
            
        self.params = []
        for layer in self.layers:
            self.params += layer.params
            
    def output(self, x): # recursively computes output for each layer
        for layer in self. layers:
            x = layer.output(x)

    def squared_error(self, x, y): # Euclidean distance between input and desired output
            return T.sum((self.output(x) - y)**2)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Cost function to minimize error and improve weights
    '''
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates


# Run

# Get and setup data - create vectors of weights, bias and activation function
W_init, b_init, activations = [], [], []

# Pass vectors into MLP and it will build the structure
mlp = MLP(W_init, b_init, activations)

# Apply gradient descent to update weights - start a learning rate and momentum
mlp_input = T.matrix('mlp_input')
mlp_target = T.vector('mlp_target')

learning_rate = 0.01
momentum = 0.9

cost = mlp.squared_error(mlp_input, mlp_target)

mlp_output = theano.function([mlp_input], mlp.output(mlp_input))