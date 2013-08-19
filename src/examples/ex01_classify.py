'''
This example demonstrates supervised backpropagation training in a simple network.
'''

# add the main source directory to the path for the imports below
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..', )))

# external libraries
import logging
import numpy as np
import random
import theano

# make sure all logging output is displayed
logging.root.setLevel(logging.DEBUG)

# causes theano to warn when a symbolic variable is not tagged with a test value.
theano.config.compute_test_value = 'warn'

# nnkit imports
from nnkit.dendrite import CompleteDendrite
from nnkit.feed import DataFeed
from nnkit.layer import InputLayer, NeuronLayer, OutputLayer
from nnkit.monitor import NetworkMonitor
from nnkit.network import Network
from nnkit.objective import ClassifyInput
from nnkit.reporter import ClassificationErrorReporter
from nnkit.sink import LoggingDataSink
from nnkit.synapse import LogisticSynapse, Synapse
from nnkit.trainer import NetworkTrainer
from nnkit.update_rule import SimpleBackprop


class BarsAndStripesTrainingDataSet(object):
    '''
    Creates a sequence of nxn (default is 4x4) matrix in which one of the rows or one of the
    columns is set to 1 and the rest of the image is 0.  Input is labeled according 0...n-1
    if row 0...n-1 is set to 1; otherwise n...2n-1 if column 0...n-1 is set to 1.
    '''
    def __init__(self, image_size=4):
        '''
        `image_size` defaults to 4.  It should be an integer >= 2.
        '''
        self.image_size = image_size


    def generate(self, N=1):
        '''
        Generate `N` random labeled bar and stripe images; return them as a list of tuples:
        (image, label).
        '''
        result = []
        for i in range(N):
            img = np.zeros((self.image_size, self.image_size))
            x = random.randint(0, 2*self.image_size-1)
            if x >= self.image_size:
                img[:, x - self.image_size] = 1
            else:
                img[x, :] = 1
                
    
            result.append((img.reshape(1, -1).astype(np.single), np.array([x], dtype=np.int64)))
        return result


def main():
    '''
    
    '''
    layers = [InputLayer(16), # 4x4 bars and stripes require inputs, one for each pixel.
              # CompleteDendrite will have 1 connection for each input-output pair; Hence the
              # weight matrix will have 256 entries, and the bias vector will have 16 elements.
              # LogisticSynapse applies the logistic function f(x) = 1/(1 + exp(-x)) to each
              # output.  SimpleBackprop as an update rule means that W <-- W - k * grad c/W, where
              # grad c/W is the gradient of the cost expression, and k is the weight learning
              # rate.  The cost expression is defined below by the ClassifyInput objective.
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), SimpleBackprop(), size=16),
              # ClassifyInput applies a softmax to its input, so a non-trivial synapse is not
              # required here.  The size of this layer should be the number of output classes
              # the input should be divided into.
              NeuronLayer(CompleteDendrite(), Synapse(), SimpleBackprop(), size=8),
              # The objective of the output layer defines the cost expression.  In this case, it
              # is uses the mean negative log likelihood (according to the model) of the correct
              # label.  This induces the network to implement logistic regression, and the argmax
              # of the output should give the network's estimate of the class of the input.
              OutputLayer(objective=ClassifyInput())]

    # When a network is constructed with a list of layers, it automatically connects the
    # predecessor successor relationships exactly as they are given in the list.  More complex
    # network geometries can be created by explicitly constructing each layer with its intended
    # predecessor.  In that case, all that the network constructor would need to receive are is
    # the input layer, eg N = Network(input_layer=L_input).
    N = Network(layers=layers)

    # Create a monitor and have it watch the output layer.
    M = NetworkMonitor().watch(layers[-1], labels=['expected_value', 'output'])

    # Compute the classification error over the last 100 items (100 is the default window size
    # when computing the moving average).

    R = ClassificationErrorReporter(feeds={ 'expected_value': DataFeed(M, layers[-1], 'expected_value'),
                                            'output':         DataFeed(M, layers[-1], 'output') },
                                    sink=LoggingDataSink())

    # Create the data generator.
    G = BarsAndStripesTrainingDataSet()

    # create the training set.
    training_set = G.generate(1500)
    validation_set = None

    # Create the trainer.
    T = NetworkTrainer(N, training_set, validation_set, batch_size=10, training_monitor=M, training_reporter=R)

    # Prepare the network for training.  This constructs and compiles the computation graph for 
    # training the network.
    T.prepare()

    # Run through one full set of training data.  Given this simple problem, it is more than
    # adequate to reach 0% error.
    T.train()

    # return the trainer.  All of the objects constructed above can be accessed through this
    # object.
    return T


if __name__ == '__main__':
    main()
