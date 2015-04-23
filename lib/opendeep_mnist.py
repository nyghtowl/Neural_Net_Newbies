'''
OpenDeep MNIST Example / Tutorial

'''
from __future__ import print_function
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.standard_datasets.image.mnist import MNIST, datasets
from opendeep.monitor.monitor import Monitor
from opendeep.monitor.plot import Plot

# set up the logger to print everything to stdout and log files in opendeep/log/logs/
config_root_logger()

def split_data(mnist_dataset):
    return mnist_dataset.getSubset(datasets.TEST)

def build_model():
    # add layers one-by-one to a Prototype container to build neural net
    # inputs_hook created automatically by Prototype; thus, no need to specify
    mlp = Prototype()
    mlp.add(BasicLayer(input_size=28*28, output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(BasicLayer(output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

    return mlp

def setup_optimization(model, n_epoch, mnist_dataset):
    # setup optimizer stochastic gradient descent 
    optimizer = SGD(model=model,
                    dataset=mnist_dataset,
                    n_epoch=n_epoch,
                    batch_size=600,
                    learning_rate=.01,
                    momentum=.9,
                    nesterov_momentum=True,
                    save_frequency=500,
                    early_stop_threshold=0.997)

    # create a Monitor to view progress on a metric other than training cost
    error = Monitor('error', model.get_monitors()['softmax_error'], train=True, valid=True, test=True)

    return optimizer, error

def evaluate(test_data, test_labels, model):
    n_examples = 50
    
    predictions = model.run(test_data[:n_examples].eval())
    actual_labels = test_labels[:n_examples].eval().astype('int32')
    # print("Predictions:", predictions)
    # print("Actual     :", actual_labels)
    print("Accuracy: ", (sum(predictions==actual_labels)*1.0/len(actual_labels)))

def main(plot=None, n_epoch=10):
    print('... loading and seting-up data')
    # don't concatenate together train and valid sets
    mnist_dataset = MNIST(concat_train_valid=False)

    print('... building the model structure')
    # create the mlp model from a Prototype
    model = build_model()
    optimizer, error = setup_optimization(model, n_epoch, mnist_dataset)
    
    print('... training the model')    
    # [optional] use keyboardInterrupt to save the latest parameters.
    if plot:
        plot = Plot("OpenDeep MLP Example", monitor_channels=error, open_browser=True)

    optimizer.train(monitor_channels=error, plot=plot)

    print('... evaluating model') 
    test_data, test_labels = split_data(mnist_dataset)
    evaluate(test_data, test_labels, model)


if __name__ == '__main__':
    main()