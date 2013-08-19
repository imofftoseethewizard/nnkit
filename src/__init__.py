'''
NNKit is a object-oriented neural network construction and exploration kit.

The following classes are largely orthogonal:

Dendrite objects specify the connectivity between a NeuronLayer and its predecessor.  Currently,
only CompleteDendrites are supported (there is a connection between every input and every output).
Eventually, this kit will have LocalDendrite, ConvolutionDendrite, and SparseDendrite.

Synapse objects wrap a differentiable function which represents the relationship between
input stimulus and output activation.  Currently, three synapse types are supported: Synapse
acts as a pass-through, f(x) = x; LogisticSynapse implements the logistic sigmoid function,
f(x) = 1/(1 + exp(-x)); and PositiveSynapse implements a negative-truncated linear function,
f(x) = x if x >= 0, f(x) = 0 otherwise.

Objective objects wrap the cost function for supervised learning.  There are currently two
objectives: BestFit and ClassifyInput.  BestFit attempts to minimize the L2 (Euclidean distance)
between the training data and the predicted output.  ClassifyInput attempts to determine the most
likely class of the input.

UpdateRule objects describe how the model parameters are updated.  Update rule has several
subclasses: SimpleBackprop provides a no-momentum, no-decay gradient descent; BackpropWithMomentum
adds a momentum term, in effect calculating an exponential moving average of the updates in
SimpleBackprop; FullBackprop extends BackpropWithMomenum with weight and bias decay.

Layer objects represent the various layers in the network.

NetworkMonitor objects extract intermediate values from a network.

Network objects aggregate Layers.

DataFeed objects encapsulate the sequence of values recorded by a NetworkMonitor taken by
particular intermediate.

DataReporter objects represent a transformation of a DataFeed into another form, the output of
which is typically directed to a DataSink.

DataSink objects encapsulate destinations of rendered data.


'''
__authors__   = "Pat Lasswell"
__copyright__ = "(c) 2013"
__license__   = "GPL v3"
__contact__   = "imofftoseethewizard@github.com"

__docformat__ = "restructuredtext en"

import component
import dendrite
import feed
import layer
import monitor
import network
import objective
import reporter
import sink
import synapse
import trainer
import update_rule
