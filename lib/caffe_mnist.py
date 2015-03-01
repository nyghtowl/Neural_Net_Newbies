'''
Caffe CNN MNIST Example / Tutorial

code from: https://groups.google.com/forum/#!topic/caffe-users/KHCU6Ti5gSQ

NOTE - below is implementing models that were already defined - need to define those models separately

use this link to define model - http://caffe.berkeleyvision.org/gathered/examples/mnist.html

Tips for how to adapt for python: https://github.com/BVLC/caffe/issues/360

'''

import sys
import caffe
import cv2
import Image
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb
caffe_root = '../'

MODEL_FILE = './examples/mnist/lenet.prototxt'
PRETRAINED = './examples/mnist/lenet_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_mode_cpu()
# Test self-made image
"""
img = caffe.io.load_image('./examples/images/two_g.jpg', color=False)
img = img.astype(np.uint8)
out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))
print out['prob'][0]
"""
db_path = '../ndsb_competition/ndsb_trial_train_lmdb'
lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0
for key, value in lmdb_cursor:
    print "Count:"
    print count
    count = count + 1
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)

    image = image.transpose()
    out = net.forward_all(data=np.asarray([image]))
    predicted_label = out['prob'][0].argmax(axis=0)
    print out['prob']
    if label == predicted_label[0][0]:
        correct = correct + 1
    print("Label is class " + str(label) + ", predicted class is " + str(predicted_label[0][0]))

print(str(correct) + " out of " + str(count) + " were classified correctly")