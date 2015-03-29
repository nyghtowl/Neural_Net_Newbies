'''
Graphlab/Dato 1 Layer Convolution Net MNIST Example / Tutorial

'''

import graphlab
import os
import tarfile

def load_data(filename):
    '''
    Graphlab stored a train/test split of the data on S3. 

    You can change the split by combing the data and splitting again.

    4 Minutes to load.
    '''
    return graphlab.SFrame(filename)
    #data['image'] = graphlab.image_analysis.resize(training_data['image'], 28, 28, 1) # potentialy need to resize data.

def create_structure():
    '''
    Tune neural net hyper parameters.

    Creates the NN model template which is similar to creating the y=mx+b model template for linear regression.

    Optional and fundamentally using convolutional neural nets either way.
    '''
    structure = graphlab.deeplearning.get_builtin_neuralnet('mnist')
    print "NN layer details: ", structure.layers 
    print "NN hyper parameters summary: ", structure.params

    return structure

def train_model(data, target, structure, max_iterations):
    '''
    Create convolutional NN 1 layer model.

    Model automatically sets validation to check model performance when training
    '''
    return graphlab.neuralnet_classifier.create(data, target=target, structure=structure, max_iterations=max_iterations)

def get_features(data):
    '''
    SArray of dense feature vectors, each of which is the concatenation of all the hidden unit values
    '''
    return model.extract_features(data)

def predict_values(model, data):
    '''
    Predict classification based on input pixels

    Input a picture of a digit and output a prediciton 
    '''
    return model.classify(data)

def evaluate_model(model, test_data):
    '''
    Error rate between predictions and labels
    '''
    eval_ = model.evaluate(test_data)
    print "Accuracy: ", eval_['accuracy']

    cf_mat = eval_['confusion_matrix']
    
    print "Confusion Matrix Correct Predictions"
    print cf_mat[cf_mat['target_label'] == cf_mat['predicted_label']].groupby('target_label', graphlab.aggregate.SUM('count')).sort('target_label')

    print "Confusion Matrix Prediction Mistakes"
    print cf_mat[cf_mat['target_label'] != cf_mat['predicted_label']].groupby('target_label', graphlab.aggregate.SUM('count')).sort('target_label')

def main(set_net_structure=True):
    '''
    Runs the full program to train the model and then evaluate
    '''
    
    train_data = load_data('http://s3.amazonaws.com/GraphLab-Datasets/mnist/sframe/train')
    test_data = load_data('http://s3.amazonaws.com/GraphLab-Datasets/mnist/sframe/test')

    if set_net_structure:
        structure = create_structure()
    else:
        structure = None

    model = train_model(train_data, 'label', structure,  3)

    evaluate_model(model, test_data)

    print "Evaluation completed"


if __name__ == '__main__':
    main()