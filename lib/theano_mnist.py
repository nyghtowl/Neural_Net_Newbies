"""
Theano Multilayer Perceptron Net MNIST Example / Tutorial

Tutorial covers logistic regression using Theano and stochastic
gradient descent optimization method.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix (W) and a bias vector (b). Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

  P(Y=i|x, W,b) = softmax_i(W*x + b) \\
                = \frac {e^{W_i*x + b_i}} {\sum_j e^{W_j*x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

  y_{pred} = argmax_i P(Y=i|x,W,b)


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class
        literall building logistic regression class


    Model coefficients/parameters are 
         weight matrix :math:`W`
        bias vector :math:`b`

    Classification is done by mapping data points onto a set of hyperplanes and the distance from the division determines class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Logistic regression parameters 

        input = theano tensor type & one minibatch
        n_in = int & # of input units 
        n_out = int & # ouptut units

        """
        # initialize weights as 0 and matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize baises b as a vector of 0s and vector of shape (n_out)
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # model structure to compute matrix of classification probabilities - depending on how many classification options there are it will return the probabilities of belonging to each class
            # W - matrix & column-k represents each class
            # x - matrix & row-j represents input training samples
            # b - vector & element-k represents free parameter of hyper plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # finds the classification with the max probability 
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters/coefficients of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """ Negative Log_likelihood Loss
        Loss function or cost function to minimize
        Mean over the dataset of individual error terms 
            distance of a prediction from target distribution
        
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        y = theano tensor type & vector of correct labels for each example

        y.shape[0] = number (n) of examples in a minibatch        
        T.arange(y.shape[0]) creates [0,1,2,... n-1] vector
        T.log(self.p_y_given_x) = log-probabilities (LP) matrix one row per example and one column per class 
        LP[T.arange(y.shape[0]),y] = minibatch vector containing [LP[0,y[0]], LP[1,y[1]], ..., - returns the log-probability of the correct label at that point in the matrix
        T.mean(LP[T.arange(y.shape[0]),y]) = the mean log-likelihood across the minibatch 
        
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Zero-One Loss 
        Loss function or cost function to minimize
        
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        y = theano tensor type & vector of correct labels for each example
        y_pred = ?

        Note: this error rate is extremely expensive to scale and thus negative log-likelihood is more prefered
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def shared_dataset(data_xy, borrow=True):
    """ Loads dataset into shared variables

    Store our dataset in shared variables to enably copying to GPU memory
    
    Break data into minibatches because copying data into the GPU is slow 
    Thus improve performance with shared variables
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, 
                            dtype=theano.config.floatX),
                            borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                            dtype=theano.config.floatX),
                            borrow=borrow)

    # GPU require float type to store data
    # Change labels to floatX
    # Need data as ints in computations becaused used as index
    # Instead of returning ``shared_y`` we will have to cast it to int. 
    # A hack to get around this issue

    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
    ''' Loads the dataset

    datset = string type and path to dataset

    train_set, valid_set, test_set = tuple(input, target) type
    input  = 2 dimension matrix numpy.ndarray & example per row
    target = 1 dimension vector numpy.ndarray with same length as # input rows

    index is used to map target to input

    '''
    # Extra code to help load file if you don't have it or its not under data dir

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    learning_rate = float offsets how much adjustment is made to weights (stochastic gradient factor)
    n_epochs = int maximal number of epochs to run the optimizer
    dataset = string MNIST dataset file path (http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz)

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28 so 784 node input and 10 node ouput
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    # 1000 epochs
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # 83 training baches = 83 loops
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    # if major improvements in error reduction, wait a lot longer before stopping
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            # patience at a min is 5000 & this determines if it will stop
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()