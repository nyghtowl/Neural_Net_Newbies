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
    # This method is to demonstrate adding layers one-by-one to a Prototype container.
    # As you can see, inputs_hook are created automatically by Prototype so we don't need to specify!
    mlp = Prototype()
    mlp.add(BasicLayer(input_size=28*28, output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(BasicLayer(output_size=512, activation='rectifier', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

    return mlp

def main():
    # grab our dataset (and don't concatenate together train and valid sets)
    mnist_dataset = MNIST(concat_train_valid=False)
    # create the mlp model from a Prototype
    mlp = create_mlp()
    # create an optimizer to train the model (stochastic gradient descent)
    optimizer = SGD(model=mlp,
                    dataset=mnist_dataset,
                    n_epoch=500,
                    batch_size=600,
                    learning_rate=.01,
                    momentum=.9,
                    nesterov_momentum=True,
                    save_frequency=500)
    # train it! feel free to do a KeyboardInterrupt - it will save the latest parameters.
    optimizer.train()

if __name__ == '__main__':
    main()