'''
Code in PyCon 2015 Presentation

'''

import graphlab

filename = 'http://s3.amazonaws.com/GraphLab-Datasets/mnist/sframe/train'

inputs = graphlab.SFrame(filename)

model = graphlab.neuralnet_classifier.create(inputs, target='label')

model.evaluate(test_data)