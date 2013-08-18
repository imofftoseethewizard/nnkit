'''
UpdateRules describe how to update the weight and bias of a dendrite to minimize the cost expression.

So far, only a few backpropagation rules are implemented.  Additional rules are planned to support
contrastive divergence training for RBMs, as well as hybrid rules for applying the negative particles
of CD/PCD as the weight decay method in traditional backprop.
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

class UpdateRule(LayerComponent):
    '''
    Abstract base class.
    '''
    pass
    
class SimpleBackprop(UpdateRule):
    '''
    Implements a simple gradient descent where weight and bias are updated by a multiple (typically small)
    of the respective cost gradient.

      W = W + lW * grad c wrt W
      b = b + lb * grad c wrt b

    where lW is the `weight_learning_weight` and lb is the `bias_learning_rate` from the
    NeruonLayer constructor.

    '''
    def weight_change(self):
        '''
        Returns an expression giving the change in weight.
        '''
        gW = self.layer.weight_gradient_expr
        lW = self.layer.weight_learning_rate

        return - lW * gW

    def bias_change(self):
        '''
        Returns an expression giving the change in bias.
        '''
        gb = self.layer.bias_gradient_expr
        lb = self.layer.bias_learning_rate

        return - lb * gb

    def updated_weight(self):
        '''
        Returns an expression giving the new weight.
        '''
        W = self.layer.weight
        dW = self.layer.weight_change_expr

        return W + dW

    def updated_bias(self):
        '''
        Returns an expression giving the new bias.
        '''
        b = self.layer.bias
        db = self.layer.bias_change_expr

        return b + db


class BackpropWithMomentum(UpdateRule):
    '''
    Implements gradient descent where weight and bias are updated by a exponential moving average
    of the cost gradient

      dW = mW * dW + (1 - mW) * grad c wrt W
      db = mb * db + (1 - mb) * grad c wrt b
      W = W + dW
      b = b + db

    where lW is the `weight_learning_weight` from the NeuronLayer constructor; similarly, lb is
    the `bias_learning_rate` parameter, mW is the `weight_momentum` parameter, and mb is the
    `bias_momenutm` parameter.
    '''
    def weight_change(self):
        '''
        Returns an expression giving the change in bias.
        '''
        dW = self.layer.weight_change
        gW = self.layer.weight_gradient_expr
        lW = self.layer.weight_learning_rate
        mW = self.layer.weight_momentum

        return mW * dW - (1 - mW) * lW * gW

    def bias_change(self):
        '''
        Returns an expression giving the change in bias.
        '''
        db = self.layer.bias_change
        gb = self.layer.bias_gradient_expr
        lb = self.layer.bias_learning_rate
        mb = self.layer.bias_momentum

        return mb * db - (1 - mb) * lb * gb


class FullBackprop(BackpropWithMomentum):
    '''
    Implements a gradient descent with both momentum in updates and decay of existing weight and bias.

      dW = mW * dW + (1 - mW) * grad c wrt W
      db = mb * db + (1 - mb) * grad c wrt b
      W = (1 - rW) * W + dW 
      b = (1 - rb) * b + db

    where lW, lb, mW, mb, rW, and rb are the `weight_learning_rate`, `bias_learning_rate`,
    `weight_momentum`, `bias_momentum`, `weight_decay_rate`, and `bias_learning_rate` parameters,
    respectively, to NeuronLayer.
    '''
    def updated_weight(self):
        '''
        Returns an expression giving the new weight.
        '''
        W = self.layer.weight
        dW = self.layer.weight_change_expr
        rW = self.layer.weight_decay

        return (1 - rW) * W + dW

    def updated_bias(self):
        '''
        Returns an expression giving the new bias.
        '''
        b = self.layer.bias
        db = self.layer.bias_change_expr
        rb = self.layer.bias_decay

        return (1 - rb) * b + db


