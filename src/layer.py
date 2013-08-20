'''
The layer module aggregates the low-level components of a neural network into basic functional units.

InputLayer and OutputLayer instances supoort input and output; OutputLayer instances also hold the
Objective component (either a BestFit or a ClassifyInput instance) which determines the cost
function to be minimized.  NeuronLayer instances perform most of the computations, each containing
a Dendrite instance to describe the connections between Layer instances, a Synapse instance to
compute the output of the layer, and and UpdateRule instance which provides the method of updating
the dendrite's weights and biases.
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


class Layer(object):
    '''
    Layer is an abstract base class.  It implements basic connectivity (predecessor/successors), and
    provides a means for NetworkMonitor instances to collect intermediate computations.
    '''

    expr_based_tap_labels = []
    var_based_tap_labels = []

    def __init__(self, size=None, batch_size=None, predecessor=None):

        self.size = size
        self.batch_size = batch_size

        self.predecessor = None
        if predecessor is not None:
            self.set_predecessor(predecessor)

        self.successors = []

        self.activation_ready = False
        self.backprop_ready = False

        # These will be lists of tuples suitable for theano.function(..., updates=<list of tuples>, ...)

        self.model_updates = []  # See NeuronLayer.prepare_backprop below.
        self.tap_updates = []    # See Layer.prepare_taps below.

        self.taps = set()


    def set_predecessor(self, l):
        '''
        Sets the predecessor to this layer.

        The predecessor's ``batch_size`` is propagated to this layer, though this does not
        automatically propagate to successors of this node.  Use ``set_batch_size`` for that.
        '''
        self.predecessor = l
        self.batch_size = l.batch_size
        l.successors.append(self)
        return self


    def set_batch_size(self, batch_size):
        '''
        Sets the batch size of this layer.  The change will propagate recursively to the layer's
        successors.  If this layer has a predecessor, the batch size of the predecessor must
        be the same as the batch size given.
        '''
        assert self.predecessor is None or self.predecessor.batch_size == batch_size
        self.batch_size = batch_size
        [l.set_batch_size(batch_size) for l in self.successors]
        return self


    # list of instance attributes containing shared variables which can be connected to the
    # layer's computation graph. This list is currently empty for all Layer subclasses except
    # NeuronLayer.
    tap_labels = []
    
    def add_tap(self, label):
        '''
        Adds a tap for an intermediate value in the layer's computation graph.  A tap is a shared
        variable used to capture the value of an expression in the computation graph.  While currently the
        only client of this method is NetworkMonitor, and the only non-trivial handler is NeuronLayer,
        it is nonetheless included here, as it is a generic facility which might have wider use in other
        layer types, e.g. a complex image processing layer like binarization.

        The ``label`` must be one of the items in the layer's ``tap_labels`` list.

        Returns a parameterless function which returns the most recent value of the tapped expression.
        '''
        assert label in self.tap_labels
        self.taps.add(label)
        return lambda: getattr(self, label).get_value()


    def prepare_taps(self):
        '''
        Computes the update pairs (shared var, expr) to supply the taps attached to this layer,
        and appends those pairs to the layer's updates list.
        '''
        assert self.backprop_ready

        for label in self.taps:
            # Some expressions associated shared variables are always updated; those that are not
            # have an expression which is updated and has the suffix _expr.  For instance, the
            # ``weight`` always has its shared variable connected to the computation graph, while
            # ``weight_gradient`` does not.  Hence ``weight_gradient`` needs to be updated by
            # ``weight_gradient_expr``.
            if label in self.expr_based_tap_labels:
                self.tap_updates.append((getattr(self, label), getattr(self, label + '_expr')))

            if label in self.var_based_tap_labels:
                self.tap_updates.append((getattr(self, label), getattr(self, label + '_var')))


    def get_parameters(self):
        '''
        Returns a dictionary which contains a selection of the layer's properties.  It is not
        intended to be used to support persistance, but as a source of diagnostic information.
        '''
        return { '__class__':  self.__class__.__name__,
                 'size':       self.size,
                 'batch_size': self.batch_size,
                 'taps':       list(self.taps),
                 }


    def prepare_activation(self):
        '''
        Constructs the computation graph for value propagation from the network input
        to the network output.

        Called after all predecessors have completed ``prepare_activation``.
        '''
        self.activation_ready = True


    def prepare_backprop(self):
        '''
        Constructs the computation graph for cost gradient and model updates.

        Called after ``prepare_activation`` and after all successors have completed
        ``prepare_backprop``.
        '''
        self.backprop_ready = True



class InputLayer(Layer):
    '''
    InputLayer instances connect input values to the computation graph.
    '''

    tap_labels = ['input',
                  'output',
                  ]

    expr_based_tap_labels = ['output']
    var_based_tap_labels = ['input']

    def prepare_activation(self):
        '''
        Creates the symbolic variable to hold a batch of input; this same variable is the layer's
        output as well.
        '''
        assert self.predecessor is None

        self.value_var = tt.matrix('x0')
        self.value_var.tag.test_value = np.random.rand(self.batch_size, self.size).astype(np.single)
        self.value = theano.shared(np.zeros((self.batch_size, self.size), dtype=np.single), 'x0')

        self.output_expr = self.value_var

        super(InputLayer, self).prepare_activation()


class OutputLayer(Layer):
    '''
    OutputLayer instances not only provide a stable endpoint where network results can be read
    from, but also contain the network Objective, a component that describes the function which
    the training process attempts to optimize.
    '''

    # These are the attribute names of shared variables that can be tapped by a NetworkMonitor.

    tap_labels = ['cost',
                  'expected_value',
                  'output',
                  ]

    expr_based_tap_labels = ['cost', 'output']
    var_based_tap_labels = ['expected_value']


    def __init__(self, objective, *args, **kwargs):
        '''
        The ``objective`` parameter should be an instance of either ``BestFit`` or
        ``ClassifyInput``.
        '''
        self.objective = objective.attach(self)
        super(OutputLayer, self).__init__(size=None, *args, **kwargs)


    def get_parameters(self):
        '''
        Returns a dictionary which contains a selection of the layer's properties.  It is not
        intended to be used to support persistance, but as a source of diagnostic information.
        '''
        d = super(OutputLayer, self).get_parameters()
        d.update({ 'objective': self.objective.get_parameters() })
        return d


    def prepare_activation(self):
        '''
        Uses the layer's objective to create a symbolic variable for the expected value, an
        expression for the network's output, and an expression of the estimated cost of the output
        (relative to the expected value).
        '''
        self.input_expr = self.predecessor.output_expr
        self.size = self.objective.size()

        self.expected_value_var = self.objective.expected_value()
        self.output_expr = self.objective.output()
        self.cost_expr = self.objective.cost()

        self.expected_value = theano.shared(self.objective.initial_expected_value(), 'z0')
        self.output = theano.shared(self.objective.initial_output(), 'z')
        self.cost = theano.shared(np.single(0), 'c')

        super(OutputLayer, self).prepare_activation()


class NeuronLayer(Layer):
    '''
    NeuronLayer instances aggregate a Dendrite instance, a Synapse instance, and an UpdateRule
    instance to produce the computational graph for forward activation and backprop model
    adjustments.  Despite being the source of the main computational tasks of this library,
    they actually contain no computation themselves, and are almost entirely plumbing.
    '''

    # These are the attribute names of shared variables that can be tapped by a NetworkMonitor.

    tap_labels = ['weight',
                  'bias',
                  'weight_change',
                  'bias_change',
                  'weight_gradient',
                  'bias_gradient',
                  'stimulus',
                  'output',
                  ]

    # The shared variables stored in these attributes are not automatically connected to the
    # computation graph, and when a tap for one of them is requested, then an additional
    # update pair will need to be generated during the ``prepare_taps`` method.

    expr_based_tap_labels = ['weight_gradient', 'bias_gradient', 'stimulus', 'output']
    
    def __init__(self, dendrite, synapse, update_rule, *args, **kwargs):
        '''
        '''

        # attach components to this layer.  See component.py.

        self.dendrite    = dendrite.attach(self)
        self.synapse     = synapse.attach(self)
        self.update_rule = update_rule.attach(self)

        super(NeuronLayer, self).__init__(*args, **kwargs)


    def get_parameters(self):
        '''
        Returns a dictionary which contains a selection of the layer's properties.  It is not
        intended to be used to support persistance, but as a source of diagnostic information.
        '''
        d = super(NeuronLayer, self).get_parameters()

        d.update({ 'dendrite':    self.dendrite.get_parameters(),
                   'synapse':     self.synapse.get_parameters(),
                   'update_rule': self.update_rule.get_parameters() })

        return d


    def prepare_activation(self):
        '''
        Constructs the computation graph for value propagation from the network input
        to the network output.

        Called after all predecessors have completed ``prepare_activation``.
        '''
        W = self.dendrite.initial_weight()
        b = self.dendrite.initial_bias()

        # Shared variables which are always updated during backprop
        self.weight        = theano.shared(W, name='W')
        self.bias          = theano.shared(b, name='b')
        self.weight_change = theano.shared(np.zeros_like(W), name='dW')
        self.bias_change   = theano.shared(np.zeros_like(b), name='db')

        # These are only part of the computation graph if they are tapped by a monitor.
        self.stimulus        = theano.shared(np.zeros((self.batch_size, self.size), dtype=np.single), name='y')
        self.output          = theano.shared(np.zeros((self.batch_size, self.size), dtype=np.single), name='z')
        self.weight_gradient = theano.shared(np.zeros_like(W), name='gW')
        self.bias_gradient   = theano.shared(np.zeros_like(b), name='gb')

        # Construct expressions to update activity
        self.input_expr = self.predecessor.output_expr
        self.stimulus_expr = self.dendrite.stimulus()
        self.output_expr = self.synapse.activity()

        super(NeuronLayer, self).prepare_activation()


    def prepare_backprop(self):
        '''
        Constructs the computation graph for cost gradient and model updates.

        Called after ``prepare_activation`` and after all successors have completed
        ``prepare_backprop``.
        '''
        assert self.successors > 0

        self.cost_expr = self.successors[0].cost_expr
        assert len(self.successors) == 1 or all(l.cost_expr == self.cost_expr for l in self.successors)
                
        self.weight_gradient_expr = self.dendrite.weight_gradient()
        self.bias_gradient_expr   = self.dendrite.bias_gradient()

        self.weight_change_expr = self.update_rule.weight_change()
        self.bias_change_expr   = self.update_rule.bias_change()

        self.updated_weight_expr = self.update_rule.updated_weight()
        self.updated_bias_expr   = self.update_rule.updated_bias()

        self.model_updates += [(self.weight,        self.updated_weight_expr),
                               (self.bias,          self.updated_bias_expr),
                               (self.weight_change, self.weight_change_expr),
                               (self.bias_change,   self.bias_change_expr)]

        super(NeuronLayer, self).prepare_backprop()

