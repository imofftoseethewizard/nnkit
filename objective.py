'''
Objective instances determine the function that the training algorithm attempts to optimize in its
changes to model parameters, and they specify the shape of the output.

There are two classes of objective: ClosestMatch and ClassifyInput.  ClosestMatch instances induce
minimization of the L2 difference (Euclidean distance) between the evaluated output and the
expected output; network output is the same as the input to the output layer containing the
ClosestMatch instance. ClassifyInput instances cause the network to use logistic regression, and
to return a single unsigned integer for each input indicating the class it most likely belongs to.
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

class Objective(LayerComponent):
    '''
    Objective is a trivial abstract base class.
    '''
    pass


class MatchInput(Objective):
    '''
    A MatchInput component attached to an output layer causes the network to implement a least
    mean squares optimization of the model weights.  The output layer's output will be a simple
    pass through from its input.
    '''

    def expected_output(self):
        '''
        Returns a symbolic variable for use as the expected output parameter to the network.  The
        variable is tagged with an appropriate test value.
        '''
        z = tt.lvector('z0')
        z.tag.test_value = np.random.randint(0, self.layer.size, (self.layer.batch_size,))
        return z


    def output(self):
        '''
        Returns the attached layer's input expression, that is, the output of the prior layer.
        '''
        return self.layer.input_expr


    def cost(self):
        '''
        Returns the mean squared difference between the output and the expected output.
        '''
        x = self.layer.input_expr
        y = self.layer.expected_value

        return tt.mean((y - x)**2)


class ClassifyInput(Objective):
    '''
    A ClassifyInput component attached to an output layer causes the network to implement logistic
    regression.  The number of classes of output is equal to the size of the input to attached
    output layer.
    '''

    def expected_value(self):
        '''
        Returns a symbolic variable for use as the expected output parameter to the network.  The
        variable is tagged with an appropriate test value.
        '''
        z = tt.lvector('z0')
        z.tag.test_value = np.random.randint(0, self.layer.size, (self.layer.batch_size,))
        return z


    def output(self):
        '''
        Returns the most likely class of the input, computed as the argmax of the softmax over of
        the attached layer's input.
        '''
        x = self.layer.input_expr
        return tt.argmax(tt.nnet.softmax(x), axis=1)


    def cost(self):
        '''
        Returns the average negative log probability of the expected output.
        '''
        x = self.layer.input_expr
        y = self.layer.expected_value

        return -tt.mean(tt.log(tt.nnet.softmax(x))[tt.arange(y.shape[0]), y])

