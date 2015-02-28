Neural_Net_Newbies
==================

Repository of different python packages' code examples to run NN on MNIST data. Some of the code here will be used in my PyCon 2015 presentation.


Theano
--------

The MNIST example is from this [link](http://deeplearning.net/tutorial/logreg.html) with some modifications especially around comments.

Data is located at this [link](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).

Additional tutorial at this [link](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb).


PyLearn2
--------

MNIST example at this [link](https://vdumoulin.github.io/articles/extending-pylearn2/) provides a solid intro to PyLearn2

Run from command line:
        python -c "from pylearn2.utils import serial; \
           train_obj = serial.load_train_file('pylearn2_log_reg.yaml'); \
           train_obj.main_loop()"


PyBrain
--------


Caffe
--------


Graphlab/Dato:
--------
MNIST sample tutorial can be found at this [link](https://dato.com/products/create/docs/graphlab.toolkits.deeplearning.html)

Image tutorial can be found at this [link](https://dato.com/learn/gallery/notebooks/build_imagenet_deeplearning.html)

Basic architectures:
    * Perceptron Network for dense numeric input
    * Convolution Network for image data input


Setup
--------
...



Resources:
--------
[Deep Learning for NLP] (http://techtalks.tv/talks/deep-learning-for-nlp-without-magic-part-1/58414/)
[Deep Learning: Machine Perception and Its Applications] (https://www.youtube.com/watch?v=hykoKDl1AtE)