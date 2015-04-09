Neural_Net_Newbies
==================

Repository of different python packages' code examples to run build, train and run a neural net on MNIST data. A small sample is used in my PyCon 2015 presentation and the rest is for reference afterwards.

[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) main site for reference.


Graphlab/Dato:
--------
MNIST sample tutorial can be found at this [link](https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html)

Image tutorial can be found at this [link](https://dato.com/learn/gallery/notebooks/build_imagenet_deeplearning.html)

Basic architectures:
    * Perceptron Network for dense numeric input
    * Convolution Network for image data input

Run from command line:
  python lib/graphlab_mnist.py

For presentation code checkout lib/slide_code.py

Setup: pip install graphlab-create & add product key to environment variable or config file

[General Reference](https://dato.com/products/create/docs/generated/graphlab.neuralnet_classifier.NeuralNetClassifier.html)

Theano
--------

The MNIST example is based off this [link](http://deeplearning.net/tutorial/logreg.html) with some modifications especially around comments.

Data is located at this [link](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).

Additional tutorial at this [link](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb).

Run from command line:
    python lib/theano_mnist.py

Setup: 
    pip install theano

[General Reference](http://deeplearning.net/software/theano/index.html)


Lasagne 
--------
MNIST example based off this [link](https://github.com/craffel/Lasagne-tutorial/blob/master/examples/mnist.py)

Machine learning library built off of Theano

Run from command line:
  python lib/lasagne_mnist.py

Setup:
    http://lasagne.readthedocs.org/en/latest/user/installation.html

[General Reference](http://lasagne.readthedocs.org/en/latest/)


Scikit-Learn
--------

MNIST example originally referenced this [link](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/) with modifications.

Note scikit-learn uses an RBM model and they are combingin the RBM with Logistict Regression to create the model that runs predictions.

Run from command line:
    python sklearn_mnist.py

Setup:
    pip install -U numpy scipy scikit-learn
    http://scikit-learn.org/stable/install.html

[General Reference](http://scikit-learn.org/stable/modules/neural_networks.html)


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

Built off of Theano

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


PyBrain
--------
MNIST example at this [link](http://martin-thoma.com/classify-mnist-with-pybrain/)

Run from command line:
    python lib/pybrain_mnist.py -e 10000 -H 300

Setup:
    pip install pybrain
    https://github.com/pybrain/pybrain/wiki/installation

[General Reference](http://pybrain.org/docs/)



Caffe
--------
MNIST example at this [link](?)


Setup:

[General Reference](http://tutorial.caffe.berkeleyvision.org/)

Setup:
--------
Most setup references assume python and pip installed. Check documentation for other options especially if setting up on GPUs.

