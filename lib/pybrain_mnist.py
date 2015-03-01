'''
PyBrain MNIST Example / Tutorial

http://martin-thoma.com/classify-mnist-with-pybrain/

'''

from struct import unpack
import gzip
from numpy import zeros, uint8, ravel

from pylab import imshow, show, cm

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from argparse import ArgumentParser
import os.path
import cPickle as pickle


def get_labeled_data(imagefile, labelfile, picklename):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        images = gzip.open(imagefile, 'rb')
        labels = gzip.open(labelfile, 'rb')

        # Read the binary data

        # We have to get big endian unsigned int. So we need '>I'

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = labels.read(4)
        N = unpack('>I', N)[0]

        if number_of_images != N:
            raise Exception('The number of labels did not match '
                            'the number of images.')

        # Get the data
        x = zeros((N, rows, cols), dtype=uint8)  # Initialize numpy array
        y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = (float(tmp_pixel) / 255)
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def classify(training, testing, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS):
    INPUT_FEATURES = testing['rows'] * testing['cols']
    print("Input features: %i" % INPUT_FEATURES)
    CLASSES = 10
    trndata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    tstdata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)

    for i in range(len(testing['x'])):
        tstdata.addSample(ravel(testing['x'][i]), [testing['y'][i]])
    for i in range(len(training['x'])):
        trndata.addSample(ravel(training['x'][i]), [training['y'][i]])

    # This is necessary, but I don't know why
    # See http://stackoverflow.com/q/8154674/562769
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,
                       outclass=SoftmaxLayer)

    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY,
                              learningrate=LEARNING_RATE,
                              lrdecay=LEARNING_RATE_DECAY)
    for i in range(EPOCHS):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])

        print("epoch: %4d" % trainer.totalepochs,
                     "  train error: %5.2f%%" % trnresult,
                     "  test error: %5.2f%%" % tstresult)
    return fnn

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=200,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-e", metavar="EPOCHS", type=int,
                        dest="epochs", default=20,
                        help="number of epochs to learn")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.01,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.1,
                        help="momentum")
    parser.add_argument("-l", metavar="ETA", type=float, dest="learning_rate",
                        default=0.01,
                        help="learning rate")
    parser.add_argument("-ld", metavar="ALPHA", type=float, dest="lrdecay",
                        default=1,
                        help="learning rate decay")
    args = parser.parse_args()

    print("Get testset")
    testing = get_labeled_data('data/pybrain/t10k-images-idx3-ubyte.gz',
                               'data/pybrain/t10k-labels-idx1-ubyte.gz', 'testing')
    print("Got %i testing datasets." % len(testing['x']))
    print("Get trainingset")
    training = get_labeled_data('data/pybrain/train-images-idx3-ubyte.gz',
                                'data/pybrain/train-labels-idx1-ubyte.gz', 'training')
    print("Got %i training datasets." % len(training['x']))
    classify(training, testing, args.hidden_neurons, args.momentum,
             args.weightdecay, args.learning_rate, args.lrdecay, args.epochs)