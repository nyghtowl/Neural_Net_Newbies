Neural_Net_Newbies
==================

Repository of machine learning python packages' code examples to build, train and run a neural net on MNIST data. A small sample is used in my PyCon 2015 presentation and the rest is for reference afterwards under lib/slide_code.py.

The "hello world" of neural nets is the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) and the link is the original source about the dataset.

Note: All examples assume supervised learning.

Theano
--------
The MNIST example in this repo is based off this [link](http://deeplearning.net/tutorial/logreg.html) with modifications.
   * Data is located at this [link](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).
   * Additional tutorial at this [link](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb).

Run code from command line:

      python lib/theano_mnist.py

Setup:

      pip install theano

[General Reference](http://deeplearning.net/software/theano/index.html) for more information about the package.

Graphlab/Dato:
--------
MNIST sample tutorial can be found at this [link](https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html). This machine learning library is built off of CXXNet.

Run code from command line:

      python lib/graphlab_mnist.py

Setup: 

      pip install graphlab-create 

   * Add product key to environment variable or config file

[General Reference](https://dato.com/products/create/docs/generated/graphlab.neuralnet_classifier.NeuralNetClassifier.html) for more information about the package.

Lasagne 
--------
The MNIST example in this repo is based off this [link](https://github.com/craffel/Lasagne-tutorial/blob/master/examples/mnist.py) with modifications. This machine learning library is built off of Theano.

Run from command line:

      python lib/lasagne_mnist.py

Setup:
   * http://lasagne.readthedocs.org/en/latest/user/installation.html

[General Reference](http://lasagne.readthedocs.org/en/latest/) for more information about the package.

PyLearn2
--------
MNIST example in this repo is based off this [link](https://vdumoulin.github.io/articles/extending-pylearn2/) with modifications. Built off of Theano and requires a yaml file to config neural net structure and optimization method.

Run from command line:

         python lib/pylearn2_mnist.py


Setup:

        git clone git://github.com/lisa-lab/pylearn2.git
        
        cd pylearn2 && python setup.py develop

        OR 

        cd pylearn2 && python setup.py develop --user

   * http://deeplearning.net/software/pylearn2/#download-and-installation
   *  You may need to add a path to the package and/or data 


[General Reference](http://deeplearning.net/software/pylearn2/) for more information.

Scikit-Learn
--------

MNIST example in this repo is based off this [link](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/) with modifications. This library uses an RBM model and they are combining the RBM with Logistict Regression to create the model that runs predictions.

Run from command line:

      python sklearn_mnist.py

Setup:

    pip install -U numpy scipy scikit-learn

   * http://scikit-learn.org/stable/install.html

[General Reference](http://scikit-learn.org/stable/modules/neural_networks.html) for more information.


Other:
--------
Most setup references assume python and pip installed. Check documentation for other options especially if setting up on GPUs.

