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
    def __init__(self, network, training_set, validation_set, batch_size,
                 protocol=None, training_monitor=None, training_reporter=None,
                 validation_interval=float('inf'), validation_monitor=None, validation_reporter=None):
        '''
        '''
        self.network = network

        self.batch_size = batch_size

        self.training_set      = training_set
        self.training_monitor  = training_monitor  or NetworkMonitor()
        self.training_reporter = training_reporter or DataReporter()
        self.training_batches  = len(self.training_set)/self.batch_size

        self.validation_set      = validation_set
        self.validation_interval = validation_interval
        self.validation_monitor  = validation_monitor  or NetworkMonitor()
        self.validation_reporter = validation_reporter or DataReporter()
        if self.validation_set is not None:
            self.validation_batches = len(self.validation_set)/self.batch_size

  

    def prepare(self):
        '''
        Prepares a network for supervised training.
        '''
        # The batch size will propagate from the input layer throughout the network.  This must be
        # done prior to network.prepare().
        self.network.input_layer.set_batch_size(self.batch_size)

        # Construct computation graphs, optimize, and compile.
        self.network.prepare_supervised_training()


    def _get_batch(self, dataset, i):
        '''
        Gets a batch of data from the given dataset.  Preps if for input for a network.
        '''
        x, z = zip(*dataset[i:i+self.batch_size])
        x = np.vstack(x)
        z = np.vstack(z)
        return x, z
        
        
    def get_training_batch(self, i):
        '''
        Get a batch of training data.
        '''
        assert i < self.training_batches
        return self._get_batch(self.training_set, i)


    def train_batch(self, i):
        '''
        Trains a network through a single batch of training data.
        '''
        self.network.train(*self.get_training_batch(i))
        self.training_monitor.collect_statistics()
        self.training_reporter.update()


    def train(self):
        '''
        Trains a network through a single epoch of training data.
        '''
        for i in range(self.training_batches):
            self.train_batch(i)
            if (i + 1) % self.validation_interval == 0:
                self.validate()


    def get_validation_batch(self, i):
        '''
        Get a batch of validation data.
        '''
        assert i < self.validation_batches
        return self._get_batch(self.validation_set, i)


    def validate_batch(self, i):
        '''
        Validates the network against a single set of validation data.
        '''
        self.network.validate(*self.get_validation_batch(i))
        self.validation_monitor.collect_statistics()
        self.validation_reporter.update()


    def validate(self):
        '''
        Validates the network against the full set of validation data.
        '''
        for i in range(self.validation_batches): self.validate_batch(i)
