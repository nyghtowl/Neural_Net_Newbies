'''
OpenDeep MNIST Example / Tutorial

'''
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.standard_datasets.image.mnist import MNIST

# set up the logger to print everything to stdout and log files in opendeep/log/logs/
config_root_logger()

def create_mlp():
    # add layers one-by-one to a Prototype container to build neural net
    # inputs_hook created automatically by Prototype; thus, no need to specify
    mlp = Prototype()
    mlp.add(BasicLayer(input_size=28*28, output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(BasicLayer(output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

    return mlp

def main():
    print '... load and setup data'
    # don't concatenate together train and valid sets
    mnist_dataset = MNIST(concat_train_valid=False)

    print '... building the model structure'
    # create the mlp model from a Prototype
    mlp = create_mlp()
    
    # setup optimizer stochastic gradient descent 
    optimizer = SGD(model=mlp,
                    dataset=mnist_dataset,
                    n_epoch=500,
                    batch_size=600,
                    learning_rate=.01,
                    momentum=.9,
                    nesterov_momentum=True,
                    save_frequency=500)

    print '... training the model'    
    # [optional] use keyboardInterrupt to save the latest parameters.
    optimizer.train()

    print '... evaluate model - TBD' 

if __name__ == '__main__':
    main()