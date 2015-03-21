Neural_Net_Newbies
==================

Repository of different python packages' code examples to run NN on MNIST data. Some of the code here will be used in my PyCon 2015 presentation.

[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) main site for reference.


Caffe
--------
MNIST example at this [link](?)


Setup:

[General Reference](http://tutorial.caffe.berkeleyvision.org/)


Graphlab/Dato:
--------
MNIST sample tutorial can be found at this [link](https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html)

Image tutorial can be found at this [link](https://dato.com/learn/gallery/notebooks/build_imagenet_deeplearning.html)

Basic architectures:
    * Perceptron Network for dense numeric input
    * Convolution Network for image data input


Setup: pip install graphlab-create

[General Reference](https://dato.com/products/create/docs/generated/graphlab.neuralnet_classifier.NeuralNetClassifier.html)

Lasagne 
--------
MNIST example at this [link](https://github.com/craffel/Lasagne-tutorial/blob/master/examples/mnist.py)

Run from command line:
  python lib/lasagne_mnist.py

Setup:
    http://lasagne.readthedocs.org/en/latest/user/installation.html

[General Reference](http://lasagne.readthedocs.org/en/latest/)



PyBrain
--------
MNIST example at this [link](http://martin-thoma.com/classify-mnist-with-pybrain/)

Run from command line:
    python lib/pybrain_mnist.py -e 10000 -H 300

Setup:
    pip install pybrain
    https://github.com/pybrain/pybrain/wiki/installation

[General Reference](http://pybrain.org/docs/)



PyLearn2
--------

MNIST example at this [link](https://vdumoulin.github.io/articles/extending-pylearn2/) provides a solid intro to PyLearn2

[dataset](http://deeplearning.net/data/mnist/mnist.pkl.gz)

Setup dataset 
        gunzip mnist.pkl.gz
        python -c "from pylearn2.utils import serial; \
           data = serial.load('mnist.pkl'); \
           serial.save('mnist_train_X.pkl', data[0][0]); \
           serial.save('mnist_train_y.pkl', data[0][1].reshape((-1, 1))); \
           serial.save('mnist_valid_X.pkl', data[1][0]); \
           serial.save('mnist_valid_y.pkl', data[1][1].reshape((-1, 1))); \
           serial.save('mnist_test_X.pkl', data[2][0]); \
           serial.save('mnist_test_y.pkl', data[2][1].reshape((-1, 1)))"

Run from command line:
        python -c "from pylearn2.utils import serial; \
           train_obj = serial.load_train_file('lib/pylearn2_log_reg.yaml'); \
           train_obj.main_loop()"


Setup:
Explanation at this [link](http://deeplearning.net/software/pylearn2/#download-and-installation)

        git clone git://github.com/lisa-lab/pylearn2.git
        cd pylearn2 && python setup.py develop

        OR 

        cd pylearn2 && python setup.py develop --user

    Note you may need to add a path to the package and/or data 


[General Reference](http://deeplearning.net/software/pylearn2/)

Scikit-Learn
--------

MNIST example is from [link](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/)

Run from command line:
    python sklearn_mnist.py --dataset data/digits.csv --test 0.4

Setup:
    pip install -U numpy scipy scikit-learn
    http://scikit-learn.org/stable/install.html

[General Reference](http://scikit-learn.org/stable/modules/neural_networks.html)

Theano
--------

The MNIST example is from this [link](http://deeplearning.net/tutorial/logreg.html) with some modifications especially around comments.

Data is located at this [link](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).

Additional tutorial at this [link](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb).

Run from command line:
    python lib/theano_mnist.py

Setup: 
    pip install theano

[General Reference](http://deeplearning.net/software/theano/index.html)


Setup:
--------
Most setup references assume python and pip installed. Check documentation for other options especially if setting up on distributed systems and GPUs.

