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

from sink import DataSink

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


class CompoundImageReporter(DataReporter):
    '''
    This class aggregates several image reporters and allows them to draw into the same output image.
    '''
    def __init__(self, reporters, result_shape, background_color=0, *args, **kwargs):
        '''
        :param reporters: is a list of ImageReporters.
        :param result_shape: is a tuple giving the output shape.
        :param background_color: gives the initial fill color of the result canvas.

        Not yet implemented: color image support.
        '''
        super(CompoundImageReporter, self).__init__(feeds=dict(), *args, **kwargs)

        self.reporters = reporters
        self.result_shape = result_shape
        self.background_color = background_color

        for r in self.reporters: r.set_parent(self)
        self.canvas = None


    def get_canvas(self):
        '''
        A new canvas is created during rendering for use by the subreporters to this one.
        '''
        return self.canvas


    def render(self, data):
        '''
        Creates a canvas for subreporters to use.  Calls update on each one, and they should each
        fill in their part of the canvas.
        '''
        self.canvas = self.background_color * np.ones(self.result_shape, dtype=np.uint8)
        for r in self.reporters: r.update()
        return self.canvas

        
class ImageReporter(DataReporter):
    pass


class ImageGridReporter(ImageReporter):
    '''
    Converts feed data into an image.
    '''
    def __init__(self, feed, image_shape, grid_shape, x0=0, y0=0, dx=None, dy=None,
                 scale=1.0, result_shape=None, *args, **kwargs):
        '''
        :param image_shape: shape of output images. feed data should be a 2d matrix, one flattened
        image per row.

        :param grid_shape: rows, columns of images.  Fills grid lexicographically (left to right,
        top to bottom).

        :param x0: left edge of image in first column of grid, defaults to 0

        :param y0: top edge of image in first row of grid, defaults to 0

        :param dx: spacing between left edges of images in each row, defaults to image_shape[1]

        :param dy: spacing between top edges of images in each column, defaults to image_shape[0]

        :param scale: scaling factor to apply to reshaped image before drawing to canvas, defaults to 1.0

        :param result_shape: the shape of the result image.  This will be ignored if parent is not 
        None; otherwise this is required.
        
        Not Implemented: color images.
        '''
        super(ImageGridReporter, self).__init__(feeds=dict(image_data=feed), *args, **kwargs)

        # these are 2-tuples
        self.image_shape  = image_shape # for reshaping each input image (pre-scaled)
        self.grid_shape   = grid_shape  # rows, cols of images
        self.result_shape = result_shape

        # these are positive integers, lengths in result image pixels
        self.x0 = x0 # left edge of image 0, 0
        self.y0 = y0 # upper edge of image 0, 0
        self.dx = dx # horizontal separation between images
        self.dy = dy # vertical separation between images

        self.scale = scale
        self.parent = None

        if self.dx is None:
            self.dx = self.image_shape[1]

        if self.dy is None:
            self.dy = self.image_shape[0]

        if self.scale < 1.0:
            self.interpolation = cv2.INTER_AREA

        elif self.scale > 1.0:
            self.interpolation = cv2.INTER_LINEAR # cv2.INTER_CUBIC would be better, but it is slower.


    def set_parent(self, parent):
        '''
        Sets the parent.  An instance with a parent will get the canvas for rendering from its
        parent, otherwise, it will create a new blank one. See get_canvas below.
        '''
        self.parent = parent
            

    def get_canvas(self):
        '''
        Get the canvas (np matrix) on which to render data.  If the instance has a parent, it will
        use the parent's canvas; otherwise, it will create a new blank canvas to use.
        '''
        if self.parent is None:
            return np.zeros(self.result_shape, dtype=np.uint8)

        return self.parent.get_canvas()


    def render_image(self, flat_image):
        '''
        Converts one row of data into an image.

        Normalizing to 0-255 by subtracting the min and scaling the max may be too heavy handed for
        some images.  Abstracting image normalization may be desirable at some point.
        '''
        img = flat_image.reshape(self.image_shape)
        if self.scale != 1.0:
            img = cv2.resize(img, 0, fx=self.scale, fy=self.scale, interpolation=self.interpolation)

        img -= img.min()
        M = img.max()
        if M > 0:
            img *= 255.99/M

        return img


    def render(self, data):
        '''
        Converts the input data into a grid of images.
        '''
        images = data['image_data']
        canvas = self.get_canvas()

        grid_height, grid_width = self.grid_shape

        x0, y0 = self.x0, self.y0
        dx, dy = self.dx, self.dy

        for i in range(images.shape[0]):
            img = self.render_image(images[i])

            c = i % grid_width
            r = i // grid_width

            if r >= grid_height:
                break

            x = dx * c + x0
            y = dy * r + y0
            
            canvas[y:y+img.shape[0], x:x+img.shape[1]] = img

        return canvas


class HistogramReporter(ImageReporter):
    '''
    Produces a histogram from the incoming data.
    '''
    def init(self, feed, bin_count=10, hist_range=None, *args, **kwargs):
        '''
        Specify the number of bins with `bin_count` (defaults to 10) and the range over which they
        will be evenly spaced with `range` (defaults to None, implying data min to data max).
        '''
        self.bin_count = bin_count
        self.hist_range = hist_range
        super(HistogramReporter, self).__init__(feeds=dict(input=feed), *args, **kwargs)


    def render(self, data):
        '''
        Use matplotlib to render data as a histogram.
        '''
        return plt.hist(data['input'], bins=self.bin_count, range=self.hist_range)


class ClassificationErrorReporter(DataReporter):
    '''
    Computes the classification error using a simple moving average.
    '''

    def __init__(self, expected_value_feed, output_feed,
                 window_size=100, label='classification error', *args, **kwargs):
        '''
        `window_size` gives the minimum number of expected/output pairs that are used in computing
        the moving average.  It may use slightly more if the batch size does not divide the window
        size.
        '''
        feeds = dict(expected_value=expected_value_feed, output=output_feed)
        super(ClassificationErrorReporter, self).__init__(feeds=feeds, *args, **kwargs)

        self.expected_value_feed = expected_value_feed
        self.window_size = window_size
        self.label = label


    def fetch(self):
        '''
        For each feed in the feeds dictionary, get all data available.
        '''
        data = dict()
        for label, feed in self.feeds.items():
            data[label] = feed.get_all()

        return data

    
    def render(self, data):
        '''
        Compute a simple moving average over the last few training batches.
        '''
        expected_value = data['expected_value']
        output = data['output']

        batch_size = output[0].size
        batches = int(math.ceil(self.window_size/batch_size))

        n = max(0, len(output) - batches)

        e = np.array(expected_value[n:]).flatten()[-self.window_size:]
        o = np.array(output[n:]).flatten()[-self.window_size:]

        return '%s: %6.2f%%' % (self.label, 100.0*np.count_nonzero(e - o)/len(o))
        
