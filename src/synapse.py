'''
A synapse is essentially a thin wrapper around a monotonic (generally) non-linear function.
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

class Synapse(LayerComponent):
    '''
    Base class. Implements f(x) = x.
    '''

    def weight_gradient(self):
        '''
        This and `bias_gradient` should probably be moved to Dendrite.
        '''
        c = self.layer.cost_expr
        W = self.layer.weight

        return tt.grad(c, W)


    def bias_gradient(self):
        '''
        This and `weight_gradient` should probably be moved to Dendrite.
        '''
        c = self.layer.cost_expr
        b = self.layer.bias

        return tt.grad(c, b)


    def activity(self):
        '''
        Returns an expression for the layer output.  In this case, it is simply the weighted and biased
        sum of the layer input.
        '''
        return self.layer.stimulus_expr


class LogisticSynapse(Synapse):
    '''
    Wraps the logistic function: f(x) = 1/(1 + exp(-x)).
    '''

    def activity(self):
        '''
        Returns an expression which computes the logistic function of the layer's stimulus expression.
        '''
        y = self.layer.stimulus_expr

        return tt.nnet.sigmoid(y)


class PositiveSynapse(Synapse):
    '''
    Wraps the function f(x) = x if x > 0, 0 otherwise.
    '''

    def activity(self):
        '''
        Returns an expression which computes (elementwise) the max of 0 and the layer's stimulus
        expression.
        
        '''
        y = self.layer.stimulus_expr

        return tt.switch(y > 0, y, 0)

