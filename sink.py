'''
DataSink instances store, display, or transmit reporter output.
'''
__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import cv2
import logging
import math
import numpy as np
import random
import os
import theano
import theano.tensor as tt

class DataSink(object):
    '''
    Base class, akin to /dev/null.
    '''
    def put_result(self, result):
        '''
        Does nothing.
        '''
        pass


class FileSystemDataSink(DataSink):
    '''
    Saves results into the file system.  

    In the root directory (given to the constructor), the data sink creates a symbolic link which
    is kept updated to the latest result.  All results are stored to unique names into a given
    subdirectory of the root directory.  The names are composed from the base name and extension
    given to the constructor, and an increasing 0-based index.
    '''
    def __init__(self, root_dir, sub_dir, base_name, extension):
        '''
        `root_dir` gives the root directory where the link will be saved.  Relative paths are
        normalized, and tildes are expanded.  In the root directory, a symbolic link will be
        created to the latest result.  The symbolic link will be named `base_name` concatenated
        with `'.'` and `extension`.  In `sub_dir` all of the results will be stored in files named
        `base_name` concatenated with `'.'`, the 0-based order of receipt, `'.'`, and `extension`.
        '''
        self.root_dir  = os.path.abspath(os.path.expanduser(root_dir))
        self.files_dir = os.path.join(self.root_dir, sub_dir)
        self.base_name = base_name
        self.extension = extension
        self.count = 0
        self.link_path = os.path.join(self.root_dir, base_name, extension)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

    def update_current_link(self):
        '''
        When a new result is received, this method removes the old symbolic link in the root
        directory and replaces it with an updated one to point to the new most recent result.
        '''
        if os.path.exists(self.link_path):
            os.remove(self.link_path)
        os.symlink(self.get_result_destination(), self.link_path)

    def get_result_destination(self):
        '''
        Returns the full path and filename to store the next result given to the data sink.
        '''
        filename = '%s.%s.%s' % (self.base_name, self.count, self.extension)
        return os.path.join(self.files_dir, filename)


    def put_result(self, result):
        '''
        Saves the result to a new file in the results subdirectory; updates the link.
        '''
        f = open(self.get_result_destination(), 'w')
        f.write(result)
        f.close()
        self.update_current_link()
        self.count += 1


class FileSystemImageSink(FileSystemDataSink):
    '''
    A subclass of FileSystemDataSink.  
    '''
    def put_result(self, result):
        '''
        Converts the result to an image before saving.  Image type is determined by the `extension`
        parameter to the constructor.
        '''
        cv2.imwrite(result, self.get_result_destination())
        self.update_current_link()
        self.count += 1


class LoggingDataSink(DataSink):
    '''
    Results are sent to the default logger.
    '''
    def __init__(self, format='%s', log_level='debug'):
        '''
        Accepts a `format` parameter suitable for interpolation with %, taking the reporter's
        result as it's only argument; default is `'%s'`.  The output log level may also be given
        in `log_level`.
        '''
        self.format = format
        self.log_level = log_level

    def put_result(self, result):
        '''
        Sends the result to the logger after interpolating into format.
        '''
        getattr(logging, self.log_level)(self.format % result)
