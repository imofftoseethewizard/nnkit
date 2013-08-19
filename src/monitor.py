'''
NetworkMonitor instances collect values from layer instances during training and use.
'''
__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import logging

class NetworkMonitor(object):
    '''
    NetworkMonitor instances collect values from network layers.  The main functions are to ensure that
    the desired values are persistent, and to enable the collection of many values from different layers
    into a dictionary with a single call.
    '''
    def __init__(self):
        '''
        This does nothing but initialize some members.
        '''
        self.reset()


    def watch(self, layer, label=None, labels=None):
        '''
        Adds a tap for the labels given.  If no labels are specified, all available taps will be
        added.
        '''
        if label is not None:
            labels = [label]

        if labels is None:
            labels = layer.tap_labels

        taps = self.taps.setdefault(layer, dict())
        stats = self.stats.setdefault(layer, dict())
            
        for label in labels:
            taps[label] = layer.add_tap(label)
            stats[label] = []

        return self


    def clear_statistics(self):
        '''
        Clears any accumulated values.  Taps remain in place.
        '''
        self.stats = dict()
        return self


    def collect_statistics(self):
        '''
        Collect the most recent value for all taps.
        '''
        if not self.paused:
            for layer, labelled_taps in self.taps.items():
                for label, tap in labelled_taps.items():
                    self.stats[layer][label].append(tap())

        return self


    def get_statistics(self, layer, label=None):
        '''
        Returns the collected values for a specific layer.  If a label is specified, then
        only those values will be returned (as a list).  If no label is specified, then 
        all collected values will be returned in a dictionary that maps each label to a
        list of values.
        '''
        if label is not None:
            return self.stats[layer][label]

        return self.stats[layer]


    def reset(self):
        '''
        Clears accumulated data and discards taps.  Note that the layers will still compute and
        save the tapped values.
        '''
        self.taps = dict()
        self.paused = False
        self.clear_statistics()
        return self


    def pause(self):
        '''
        Pause data collection. When paused, calls to ``collect_statistics`` do nothing.
        '''
        self.paused = True
        return self


    def resume(self):
        '''
        Resume data collection.
        '''
        self.paused = False
        return self
