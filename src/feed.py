'''
The feed module contains a single simple class: DataFeed.
'''

__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import logging
import numpy as np

class DataFeed(object):
    '''
    DataFeeds provide a simple handle to the sequence of values associated with an intermediate
    computation as recorded by a NetworkMonitor.
    '''
    def __init__(self, monitor, layer, label):
        '''
        Creates a new data feed based on the *label* data collected by *monitor* from *layer*.

        Example::
            L = NeuronLayer(...)
            ...
            M = Monitor().add_subject(L)
            F = DataFeed(M, L, 'weight_gradient')
        '''
        self.monitor = monitor
        self.layer = layer
        self.label = label
        
        self.monitor.watch(layer, label=label)


    def get_latest(self):
        '''
        Returns the latest value recorded by the associated monitor.
        '''
        return self.monitor.get_statistics(self.layer, self.label)[-1]


    def get_all(self):
        '''
        Returns all values recorded by the associated monitor.
        '''
        return self.monitor.get_statistics(self.layer, self.label)


class TailingDataFeed(DataFeed):
    '''
    '''
    def __init__(self, monitor, layer, label, tail_length=1):
        '''
        '''
        super(TailingDataFeed, self).__init__(monitor, layer, label)
        self.tail_length = tail_length


    def get_latest(self):
        '''
        Returns a slice of length tail_length from the collected data.

        '''
        return np.vstack(super(TailingDataFeed, self).get_all()[-self.tail_length:])
