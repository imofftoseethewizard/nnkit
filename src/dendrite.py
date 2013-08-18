'''
Dendrite components provide the connectivity between network layers.  

There are four Dendrite subclasses: CompleteDendrite, SparseDendrite, LocalDendrite, and
ConvolutionDendrite.

CompleteDendrite instances provide a connection from each input to each output.

SparseDendrite instances provide a connection between only some of the inputs and some of the
outputs.  In general, the connectivity will be irregular, though this is by no means required. In
fact, both the LocalDendrite and ConvolutionDendrite have a regular connection geometry and are
implemented as SparseDendrite subclassses.

In LocalDendrite instances, outputs only have connections to inputs which are in some sense near
to them.

ConvolutionDendrite is similar to LocalDendrite, except that every output has the same
connectivity and weights, modulo translation.

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

from component import LayerComponent

class Dendrite(LayerComponent):
    '''
    Dendrite is an abstract base class, and a trivial subclass of LayerComponent.
    '''
    pass


class CompleteDendrite(Dendrite):
    '''
    CompleteDendrite provides a connection from every input to every output. 

    Example:
        L = NeuronLayer(CompleteDendrite(), ...)

    '''
    def initial_weight(self):
        '''
        Returns a numpy array of small random numbers of size appropriate for use as a weight
        matrix.
        '''

        # This formula is from "Understanding the difficulty of training deep feedforward neural
        # networks" by Xavier Glorot and Yoshua Bengio.  Ungated paper available at
        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf

        rows = self.layer.predecessor.size
        cols = self.layer.size

        w = math.sqrt(6.0/(rows + cols))
        return np.random.uniform(-w, w, (rows, cols)).astype(np.single)

                                  
    def initial_bias(self):
        '''
        Returns a numpy array of zeros of size appropriate for use as a bias vector.
        '''
        return np.zeros((1, self.layer.size), dtype=np.single)

    def stimulus(self):
        '''
        Computes the pre-synaptic output given the layer's input. It computes

          y = xW + b

        where x is the output of the prior layer, W is the weight matrix, and b is the bias vector.
        '''
        x = self.layer.input_expr
        W = self.layer.weight
        b = self.layer.bias

        return x.dot(W) + tt.addbroadcast(b, 0)


class SparseDendrite(Dendrite):
    '''    '''
    def get_layer_stimulus(self, layer):
        b = layer.get_biases()
        x = layer.get_input()
        y = layer.get_stimulus()
        return theano.function([], y, updates=[(y, b)])


class LocalDendrite(SparseDendrite):
    # each dendrite only connects to a small area of the prior layer
    pass

class ConvolutionDendrite(Dendrite):
    # each dendrite only connects to a small area of the prior layer
    # similar to LocalDendrite, but with shared weights
    pass


