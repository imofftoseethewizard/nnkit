'''
Network instances create the training and evaluation functions of a stack of layers.
'''
__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import logging
import theano
import theano.tensor as tt

from layer import OutputLayer
        
class Network(object):
    '''
    Network instances build the training and evaluation computations of a stack of network layers.
    
    At present, only supervised backpropagation is a supported training method.  In the future
    networks will support unsupervised training as a stack of restricted Boltzmann machines (a
    deep belief network).  
    '''
    def __init__(self, layers=None, input_layer=None):
        '''
        If ``layers`` is specified, then it must be a list of layers, beginning with an input
        layer, and ending with an output layer, and in this case the ``input_layer`` parameter
        will be ignored.  The predecessor/successor relationships of the layers in the list will
        be changed to reflect the order of the layers in the list.

        Otherwise, the network will start with the given ``input_layer``, and it is presumed that
        the predecessor/successor relationships have already been built and that the last
        layer in 
        '''
        self.input_layer = input_layer
        self.output_layer = None

        if layers is not None:
            self.input_layer = layers[0]
            self.output_layer = layers[-1]

            [s.set_predecessor(p) for p, s in zip(layers[:-1], layers[1:])]


    def prepare_unsupervised_training(self):
        '''
        Not yet implemented.

        Prepare to train network as a deep belief network where each layer is trained in turn from
        input to output as a restricted Boltzmann machine using some flavor of contrastive
        divergence.
        '''
        pass


    def prepare_supervised_training(self, enableTaps=True):
        '''
        Prepares the network for training.  Creates a ``evaluate`` method which takes an input
        batch and returns the result of applying the network to it; also creates a ``train``
        method which takes a batch of input and a batch of expected output and returns the same results
        as ``evalutate`` plus the cost (relative to the expected output) of that result.

        To disable instrumentation, pass ``enableTaps`` as False.

        '''
        # recompute layers from input layer, find output layer, and construct computation for
        # activation.

        self.prepare_evaluation()

        # construct computation for cost gradient vs weight and bias, and for weight and bias
        # updates.

        for l in reversed(self.layers): l.prepare_backprop()

        # ensure that monitored values are connected to the computation graph.

        if enableTaps:
            for l in self.layers: l.prepare_taps()

        # create training function.  It accepts as arguments one batch of training input, and one
        # batch of expected output; it returns a batch of resulting output, and the cost
        # associated with that batch.

        self.train = theano.function(inputs=(self.input_layer.value_var, self.output_layer.expected_value_var),
                                     outputs=(self.output_layer.output_expr, self.output_layer.cost_expr),
                                     updates=sum((l.tap_updates + l.model_updates for l in self.layers), []))

        # create validation function.  This differs from the training function only in that it does not
        # update the any of the model parameters.

        self.validate = theano.function(inputs=(self.input_layer.value_var, self.output_layer.expected_value_var),
                                        outputs=(self.output_layer.output_expr, self.output_layer.cost_expr),
                                        updates=sum((l.tap_updates for l in self.layers), []))


    def prepare_evaluation(self):
        '''
        Prepares the network for evaluating input.  This will create a static network with
        no backpropagation.  This method will add an ``eval`` method to its instance.
        '''
        layers = [self.input_layer]
        idx = 0
        while idx < len(layers):
            l = layers[idx]
            idx += 1
            if not l.activation_ready:
                l.prepare_activation()
                layers.extend(l.successors)
                if isinstance(l, OutputLayer):
                    self.output_layer = layers[-1]

        self.layers = layers

        self.eval = theano.function(inputs=(self.input_layer.value_var,),
                                    outputs=(self.output_layer.output_expr,))
                                    

