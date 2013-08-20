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
from nnkit.feed import DataFeed, TailingDataFeed
from nnkit.layer import InputLayer, NeuronLayer, OutputLayer
from nnkit.monitor import NetworkMonitor
from nnkit.network import Network
from nnkit.objective import BestFit
from nnkit.reporter import CompoundImageReporter, ImageGridReporter
from nnkit.sink import FileSystemImageSink
from nnkit.synapse import LogisticSynapse, PositiveSynapse, Synapse
from nnkit.trainer import NetworkTrainer
from nnkit.update_rule import BackpropWithMomentum, FullBackprop, SimpleBackprop


class CirclesTrainingDataSet(object):
    '''
    Creates a sequence of nxn (default is 16x16) matrices which contain a circle of fixed radius
    (default is 3 pixels) at a random location.  There is a small amount of gaussian blur (default
    sigma is 1.5 pixels) and there is gaussian noise (default sigma is 20% of max intensity).
    Labels are the images prior to noise being added.
    '''
    def __init__(self, image_size=16, radius=3, blur=1.5, noise=0.2):
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
            label = np.zeros((self.image_size, self.image_size))

            x = random.randrange(self.radius+1, self.image_size-self.radius-1)
            y = random.randrange(self.radius+1, self.image_size-self.radius-1)
            cv2.circle(label, (x, y), self.radius, 1, -1)

            label = cv2.GaussianBlur(label, (0, 0), self.blur)
            img = label + np.random.normal(0, self.noise, (self.image_size, self.image_size))

            img = img.astype(np.single).reshape(1, -1)
            label = label.astype(np.single).reshape(1, -1)

            result.append((img, label))
        return result


def main():
    '''
    '''
    # The set-up for the reporter is cumbersome.
    # TODO: Make it not so.
    image_size = 24
    margin = 10
    image_spacing = 5
    column_spacing = 15

    input_layer   = InputLayer(image_size**2)
    neuron_layer0 = NeuronLayer(CompleteDendrite(), PositiveSynapse(), SimpleBackprop(), image_size**2)
    neuron_layer1 = NeuronLayer(CompleteDendrite(), LogisticSynapse(), SimpleBackprop(), image_size**2)
    output_layer  = OutputLayer(objective=BestFit())

    # When a network is constructed with a list of layers, it automatically connects the
    # predecessor successor relationships exactly as they are given in the list.  More complex
    # network geometries can be created by explicitly constructing each layer with its intended
    # predecessor.  In that case, all that the network constructor would need to receive are is
    # the input layer, eg N = Network(input_layer=L_input).

    N = Network(layers=[input_layer, neuron_layer0, neuron_layer1, output_layer])

    # Create the data generator.
    D = CirclesTrainingDataSet(image_size=image_size, radius=5, noise=0.1)

    # create the training set.
    training_set = D.generate(10000)
    validation_set = D.generate(10)
    
    M = NetworkMonitor()

    dx = 2*(image_size + image_spacing) + image_size + column_spacing
    dy = image_size + image_spacing

    reporters = [ImageGridReporter(image_shape=(image_size, image_size), grid_shape=(5, 2),
                                   x0=margin, y0=margin,
                                   dx=dx, dy=dy,
                                   feed=TailingDataFeed(M, input_layer, 'output', tail_length=10)),
                 ImageGridReporter(image_shape=(image_size, image_size), grid_shape=(5, 2),
                                   x0=margin + image_size + image_spacing, y0=margin,
                                   dx=dx, dy=dy,
                                   feed=TailingDataFeed(M, output_layer, 'expected_value', tail_length=10)),
                 ImageGridReporter(image_shape=(image_size, image_size), grid_shape=(5, 2),
                                   x0=margin + 2*(image_size + image_spacing), y0=margin,
                                   dx=dx, dy=dy,
                                   feed=TailingDataFeed(M, output_layer, 'output', tail_length=10))]

    image_sink = FileSystemImageSink(root_dir=os.path.join(os.getcwd(), 'ex02_images'),
                                     sub_dir='all',
                                     base_name='output_comparison',
                                     extension='png')

    result_shape = (2*margin + 5*image_size + 4*image_spacing, 2*margin + 6*image_size + 4*image_spacing + column_spacing)
    R = CompoundImageReporter(reporters, result_shape, background_color=128, sink=image_sink)

    # Create the trainer.
    T = NetworkTrainer(N, training_set, validation_set, batch_size=1, validation_interval=100,
                       validation_monitor=M, validation_reporter=R)

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
