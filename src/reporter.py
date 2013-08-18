'''
Reporter instances collect data from one or more DataFeeds, process it into another form -- an
image, a histogram, a text-log, etc -- and then deliver it to a DataSink.
'''

__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import theano
import theano.tensor as tt


class DataReporter(object):
    '''
    This is the base class of all other reporters.  It sends to the sink the data exactly as
    collected from its feeds.
    '''
    def __init__(self, feeds=dict(), sink=None):
        '''
        The basic data reporter takes a dictionary of feeds and a sink.  If no sink is provided
        then a default (null) sink is created.
        '''
        self.feeds = dict(feeds)

        self.sink = sink
        if sink is None:
            self.sink = DataSink()


    def update(self):
        '''
        Fetches the latest data from the feeds, renders it into output, and emits it.
        '''
        data = self.fetch()
        result = self.render(data)
        self.emit(result)


    def fetch(self):
        '''
        For each feed in the feeds dictionary, this get the latest value, and fills a parallel
        dictionary with the latest values keyed by the key of the respective source feed.
        '''
        data = dict()
        for label, feed in self.feeds.items():
            data[label] = feed.get_latest()

        return data

    
    def render(self, data):
        '''
        In the base class, this simply returns its argument.  Derived classes should implement
        their own version of this function.
        '''
        return data


    def emit(self, result):
        '''
        Puts the rendered result into the reporters data sink.
        '''
        self.sink.put_result(result)

        
class ImageReporter(DataReporter):
    '''
    Converts feed data into an image.
    '''
    def __init__(self):
        '''
        Not yet implemented.

        This constructor should take parameters describing how the image should be created: size,
        pixel depth, format, post-processing, etc.
        '''
        pass


    def render(self, data):
        '''
        Not yet implemented.

        Converts the input data into an image.
        '''
        return data


class HistogramReporter(ImageReporter):
    '''
    Produces a histogram from the incoming data.
    '''
    def init(self, bin_count=10, hist_range=None, *args, **kwargs):
        '''
        Specify the number of bins with `bin_count` (defaults to 10) and the range over which they
        will be evenly spaced with `range` (defaults to None, implying data min to data max).
        '''
        self.bin_count = bin_count
        self.hist_range = hist_range
        super(HistogramReporter, self).__init__(*args, **kwargs)

    def render(self, data):
        '''
        Use matplotlib to render data as a histogram.
        '''
        return plt.hist(data, bins=self.bin_count, range=self.hist_range)

