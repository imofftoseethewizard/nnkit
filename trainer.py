'''
NetworkTrainer instances are largely just a stub.  For now, they prepare a network for training,
monitor it during training, and report the results.  In the future, these will measure the
progress of training and alter the learning parameters, the training protocol, or the network
geometry in an attempt to improve network accuracy.
'''

__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import logging
import math
import numpy as np
import random
import theano
import theano.tensor as tt

from monitor import NetworkMonitor
from reporter import DataReporter

class NetworkTrainer(object):
    '''
    A simple class that prepares a network, trains it, and reports the results.
    '''
    def __init__(self, network, training_set, validation_set,
                 batch_size, protocol=None, monitor=None, reporter=None):
        '''
        `protocol` is currently unused, but is intended to supply parameters to the
        training method.
        '''
        self.network = network
        self.batch_size = batch_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.protocol = protocol
        self.monitor = monitor or NetworkMonitor()
        self.reporter = reporter or DataReporter()


    def prepare(self):
        '''
        Prepares a network for supervised training.
        '''
        # The batch size will propagate from the input layer throughout the network.  This must be
        # done prior to network.prepare().
        self.network.input_layer.set_batch_size(self.batch_size)

        # Construct computation graphs, optimize, and compile.
        self.network.prepare_supevised_training()


    def train(self):
        '''
        Trains a network through a single epoch of training data.
        '''
        batches = len(self.training_set)/self.batch_size
        for i in range(0, len(self.training_set), self.batch_size):
            x, z = zip(*self.training_set[i:i+self.batch_size])
            y, c = self.network.train(np.vstack(x), np.array(z))
            self.monitor.collect_statistics()
            logging.debug('actual: %s' % (y, ))
            logging.debug('expected: %s' % (z, ))
            logging.debug('cost: %s' % (c, ))
        #if self.reporter:
         #       self.reporter.update()
