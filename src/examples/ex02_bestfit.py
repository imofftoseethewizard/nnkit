'''
This example demonstrates supervised backpropagation training in a simple network which approximates
the training set.
'''

# add the main source directory to the path for the imports below
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..', )))

# external libraries
import cv2
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
from nnkit.layer import InputLayer, NeuronLayer, OutputLayer
from nnkit.network import Network
from nnkit.objective import BestFit
from nnkit.synapse import LogisticSynapse, Synapse
from nnkit.trainer import NetworkTrainer
from nnkit.update_rule import BackpropWithMomentum


class CirclesTrainingDataSet(object):
    '''
    Creates a sequence of nxn (default is 16x16) matrices which contain a circle of fixed radius
    (default is 3 pixels) at a random location.  There is a small amount of gaussian blur (default
    sigma is 1.5 pixels) and there is gaussian noise (default sigma is 10% of blurred pixel).
    This data is labeled with nxn matrices where all pixels are zero except for the one in containing
    the center of the circle in the input image.
    '''
    def __init__(self, image_size=16, radius=3, blur=1.5, noise=0.1):
        self.image_size = image_size
        self.radius = radius
        self.blur = blur
        self.noise = noise

    def generate(self, N=1):
        '''
        Generate `N` random labeled noisy circle images; return them as a list of tuples:
        (image, label).
        '''
        result = []
        for i in range(N):
            img = np.zeros((self.image_size, self.image_size))

            x = random.randrange(0, self.image_size)
            y = random.randrange(0, self.image_size)
            cv2.circle(img, (x, y), self.radius, 1, -1)

            img = cv2.GaussianBlur(img, (0, 0), self.blur)
            img *= np.random.normal(0, self.noise, (self.image_size, self.image_size))

            img = np.where(0 < img, np.where(img < 1, img, 1), 0).astype(np.single)
            label = np.zeros((self.image_size, self.image_size), dtype=np.single)
            label[y, x] = 1

            result.append((img.reshape(1, -1), label.reshape(1, -1)))
        return result


def main():
    '''
    
    '''
    layers = [InputLayer(256), # 16x16 noisy circles require inputs, one for each pixel.
              # CompleteDendrite will have 1 connection for each input-output pair; Hence the
              # weight matrix will have 65536 entries, and the bias vector will have 256 elements.
              # LogisticSynapse applies the logistic function f(x) = 1/(1 + exp(-x)) to each
              # output.  SimpleBackprop as an update rule means that W <-- W - k * grad c/W, where
              # grad c/W is the gradient of the cost expression, and k is the weight learning
              # rate.  The cost expression is defined below by the ClassifyInput objective.
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), BackpropWithMomentum(), size=256),
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), BackpropWithMomentum(), size=256),
              OutputLayer(objective=BestFit())]

    # When a network is constructed with a list of layers, it automatically connects the
    # predecessor successor relationships exactly as they are given in the list.  More complex
    # network geometries can be created by explicitly constructing each layer with its intended
    # predecessor.  In that case, all that the network constructor would need to receive are is
    # the input layer, eg N = Network(input_layer=L_input).
    N = Network(layers=layers)

    # Create the data generator.
    G = CirclesTrainingDataSet()

    # create the training set.
    training_set = G.generate(1000)
    validation_set = None

    # Create the trainer.
    T = NetworkTrainer(N, training_set, validation_set, batch_size=10)

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
