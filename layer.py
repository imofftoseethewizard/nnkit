import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import theano
import theano.tensor as tt

logging.root.setLevel(logging.DEBUG)
theano.config.compute_test_value = 'warn'

#TODO: image reporter
#  image shape x,y or x, y, 3
#TODO: reporting
#  use matplotlib
#  image
#  histogram
#  heatmap
#  histogram/image/heatmap sequence
#  movie (sequence of images/histograms) cv2.VideoWriter
#  histogram of activations
#  histogram of weights
#  histogram of weight delta
#  histogram of biases
#  histogram of bias delta
#  dump to files
#  dump directory
#  maintain symbolic link to latest
#TODO: sampling layer
#TODO: abstract connection pattern
#  dendrite
#    complete 
#    sparse
#    local
#    convolution
# TODO: additional synapse types:
#  linear synapse (for completeness)
#  soft positive -- log (1 + e^x)
#  tanh
#TODO: modulation layers
# feedback into bias
# dendrite type?
# feedback into weights?
#TODO: verification tests
#TODO: batch mode
#TODO: serialization
#TODO: sampling layer
#TODO: convolution layer
#TODO: dropout


class LayerComponent(object):
    def attach(self, layer):
        self.layer = layer
        return self

class Dendrite(LayerComponent):
    pass


class CompleteDendrite(Dendrite):
    def initial_weight(self):

        # This formula is from "Understanding the difficulty of training deep feedforward neural
        # networks" by Xavier Glorot and Yoshua Bengio.  Ungated paper available at
        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf

        rows = self.layer.predecessor.size
        cols = self.layer.size

        w = math.sqrt(6.0/(rows + cols))
        return np.random.uniform(-w, w, (rows, cols)).astype(np.single)

                                  
    def initial_bias(self):
        return np.zeros((1, self.layer.size), dtype=np.single)

    def stimulus(self):
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


class LocalDendrite(Dendrite):
    # each dendrite only connects to a small area of the prior layer
    pass

class ConvolutionDendrite(Dendrite):
    # each dendrite only connects to a small area of the prior layer
    # similar to LocalDendrite, but with shared weights
    pass


class Synapse(LayerComponent):

    def weight_gradient(self):
        c = self.layer.cost_expr
        W = self.layer.weight

        return tt.grad(c, W)

    def bias_gradient(self):
        c = self.layer.cost_expr
        b = self.layer.bias

        return tt.grad(c, b)


class LogisticSynapse(Synapse):

    def activity(self):
        y = self.layer.stimulus_expr

        return tt.nnet.sigmoid(y)

class PositiveSynapse(Synapse):

    def activity(self):
        y = self.layer.stimulus_expr

        return tt.switch(y > 0, y, 0)


class Objective(LayerComponent):
    pass


class MatchInput(Objective):
# TODO:
    
    def expected_output(self):
        z = tt.lvector('z0')
        z.tag.test_value = np.random.randint(0, self.layer.size, (self.layer.batch_size,))
        return z

    def output(self):
        x = self.layer.input_expr
        return tt.argmax(tt.nnet.softmax(x), axis=1)

    def cost(self):
        x = self.layer.input_expr
        y = self.layer.expected_value

        return -tt.mean(tt.log(tt.nnet.softmax(x))[tt.arange(y.shape[0]), y])

#        return theano.function([], d, updates=[(d, (z - x).T)])


class ClassifyInput(Objective):

    def expected_value(self):
        z = tt.lvector('z0')
        z.tag.test_value = np.random.randint(0, self.layer.size, (self.layer.batch_size,))
        return z

    def output(self):
        x = self.layer.input_expr
        return tt.argmax(tt.nnet.softmax(x), axis=1)

    def cost(self):
        x = self.layer.input_expr
        y = self.layer.expected_value

        return -tt.mean(tt.log(tt.nnet.softmax(x))[tt.arange(y.shape[0]), y])


class UpdateRule(LayerComponent):
    pass
    
class SimpleBackprop(UpdateRule):
    
    def weight_change(self):
        gW = self.layer.weight_gradient_expr
        lW = self.layer.weight_learning_rate

        return - lW * gW

    def bias_change(self):
        gb = self.layer.bias_gradient_expr
        lb = self.layer.bias_learning_rate

        return - lb * gb

    def updated_weight(self):
        W = self.layer.weight
        dW = self.layer.weight_change_expr

        return W + dW

    def updated_bias(self):
        b = self.layer.bias
        db = self.layer.bias_change_expr

        return b + db


class BackpropWithMomentum(UpdateRule):
    
    def weight_change(self):
        dW = self.layer.weight_change
        gW = self.layer.weight_gradient_expr
        lW = self.layer.weight_learning_rate
        mW = self.layer.weight_momentum

        return mW * dW - (1 - mW) * lW * gW

    def bias_change(self):
        db = self.layer.bias_change
        gb = self.layer.bias_gradient_expr
        lb = self.layer.bias_learning_rate
        mb = self.layer.bias_momentum

        return mb * db - (1 - mb) * lb * gb


class FullBackprop(BackpropWithMomentum):
    
    def updated_weight(self):
        W = self.layer.weight
        dW = self.layer.weight_change_expr
        rW = self.layer.weight_decay

        return (1 - rW) * W + dW

    def updated_bias(self):
        b = self.layer.bias
        db = self.layer.bias_change_expr
        rb = self.layer.bias_decay

        return (1 - rb) * b + db


class Layer(object):
    def __init__(self, size, batch_size=None, predecessor=None, weight_learning_rate=0.2, bias_learning_rate=0.2,
                 weight_momentum=0.5, bias_momentum=0.5, weight_decay=0.0, bias_decay=0.0):

        self.size = size
        self.batch_size = batch_size

        self.predecessor = None
        if predecessor is not None:
            self.set_predecessor(predecessor)

        self.successors = []

        self.weight_learning_rate = theano.shared(np.single(weight_learning_rate), 'lW')
        self.bias_learning_rate   = theano.shared(np.single(bias_learning_rate), 'lb')

        self.weight_momentum = theano.shared(np.single(weight_momentum), 'mW')
        self.bias_momentum   = theano.shared(np.single(bias_momentum), 'mb')

        self.weight_decay = theano.shared(np.single(weight_decay), 'rW')
        self.bias_decay   = theano.shared(np.single(bias_decay), 'rb')

        self.activation_ready = False
        self.backprop_ready = False
        self.updates = []
        self.taps = set()


    def set_predecessor(self, l):
        self.predecessor = l
        self.batch_size = l.batch_size
        l.successors.append(self)
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        [l.set_batch_size(batch_size) for l in self.successors]
        return self

    valid_tap_labels = ['weight',
                        'bias',
                        'weight_change',
                        'bias_change',
                        'weight_gradient',
                        'bias_gradient',
                        'stimulus',
                        'output',
                        ]

    
    def add_tap(self, label):
        '''
        A tap is a shared variable used to capture the value of expressions in the computation graph.
        '''
        assert label in Layer.valid_tap_labels
        self.taps.add(label)
        return lambda: getattr(self, label).get_value()

    def prepare_taps(self):
        '''
        Computes the update pairs (shared var, expr) to supply the taps attached to this layer.
        '''
        for label in self.taps:
            if label in ['weight_gradient', 'bias_gradient', 'stimulus', 'output']:
                self.updates.append((getattr(self, label), getattr(self, label + '_expr')))

    def get_parameters(self):
        return { 'weight_learning_rate': self.weight_learning_rate.get_value(),
                 'weight_momentum':      self.weight_momentum.get_value(),
                 'weight_decay':         self.weight_decay.get_value(),
                 'bias_learning_rate':   self.bias_learning_rate.get_value(),
                 'bias_momentum':        self.bias_momentum.get_value(),
                 'bias_decay':           self.bias_decay.get_value() }


    def prepare_activation(self):
        self.activation_ready = True


    def prepare_backprop(self):
        # add any updates required by monitor taps.
        self.prepare_taps()

        self.backprop_ready = True



class InputLayer(Layer):
    def prepare_activation(self):
        self.value = tt.matrix('x0')
        self.value.tag.test_value = np.random.rand(self.batch_size, self.size).astype(np.single)
        self.output_expr = self.value

        super(InputLayer, self).prepare_activation()


class OutputLayer(Layer):
    def __init__(self, objective, *args, **kwargs):
        self.objective = objective.attach(self)
        super(OutputLayer, self).__init__(*args, **kwargs)

    def prepare_activation(self):
        self.input_expr = self.predecessor.output_expr

        self.expected_value = self.objective.expected_value()
        self.output_expr = self.objective.output()
        self.cost_expr = self.objective.cost()

        super(OutputLayer, self).prepare_activation()


class NeuronLayer(Layer):
    def __init__(self, dendrite, synapse, update_rule, *args, **kwargs):

        self.dendrite    = dendrite.attach(self)
        self.synapse     = synapse.attach(self)
        self.update_rule = update_rule.attach(self)

        super(NeuronLayer, self).__init__(*args, **kwargs)


    def prepare_activation(self):
        self.input_expr = self.predecessor.output_expr

        W = self.dendrite.initial_weight()
        b = self.dendrite.initial_bias()

        self.weight = theano.shared(W, name='W')
        self.bias   = theano.shared(b, name='b')

        self.weight_change = theano.shared(np.zeros_like(W), name='dW')
        self.bias_change   = theano.shared(np.zeros_like(b), name='db')

        self.stimulus_expr = self.dendrite.stimulus()
        self.output_expr = self.synapse.activity()

        # These are only part of the computation graph if they are tapped by a monitor.
        self.stimulus        = theano.shared(np.zeros((self.batch_size, self.size), dtype=np.single), name='y')
        self.output          = theano.shared(np.zeros((self.batch_size, self.size), dtype=np.single), name='z')
        self.weight_gradient = theano.shared(np.zeros_like(W), name='gW')
        self.bias_gradient   = theano.shared(np.zeros_like(b), name='gb')

        super(NeuronLayer, self).prepare_activation()


    def prepare_backprop(self):
        assert self.successors > 0

        self.cost_expr = self.successors[0].cost_expr
        assert len(self.successors) == 1 or all(l.cost_expr == self.cost_expr for l in self.successors)
                
        self.weight_gradient_expr = self.synapse.weight_gradient()
        self.bias_gradient_expr   = self.synapse.bias_gradient()

        self.weight_change_expr = self.update_rule.weight_change()
        self.bias_change_expr   = self.update_rule.bias_change()

        self.updated_weight_expr = self.update_rule.updated_weight()
        self.updated_bias_expr   = self.update_rule.updated_bias()

        self.updates += [(self.weight,        self.updated_weight_expr),
                         (self.bias,          self.updated_bias_expr),
                         (self.weight_change, self.weight_change_expr),
                         (self.bias_change,   self.bias_change_expr)]

        super(NeuronLayer, self).prepare_backprop()


        
class Network(object):
    def __init__(self, layers=None, input_layer=None, output_layer=None):
        self.input_layer = input_layer
        self.output_layer = output_layer

        if layers is not None:
            self.input_layer = layers[0]
            self.output_layer = layers[-1]

            [s.set_predecessor(p) for p, s in zip(layers[:-1], layers[1:])]


    def prepare(self):
        layers = [self.input_layer]
        idx = 0
        while idx < len(layers):
            l = layers[idx]
            idx += 1
            if not l.activation_ready:
                l.prepare_activation()
                layers.extend(l.successors)

        self.layers = layers
        for l in reversed(self.layers): l.prepare_backprop()

        self.updates = sum((l.updates for l in layers), [])

        self.train = theano.function(inputs=(self.input_layer.value, self.output_layer.expected_value),
                                     outputs=(self.output_layer.output_expr, self.output_layer.cost_expr),
                                     updates=self.updates)

        self.test = theano.function(inputs=(self.input_layer.value,),
                                    outputs=(self.output_layer.output_expr,))
                                    

                

class NetworkTrainer(object):
    def __init__(self, network, training_set, validation_set, batch_size, protocol=None, monitor=None, reporter=None):
        self.network = network
        self.batch_size = batch_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.protocol = protocol
        self.monitor = monitor or NetworkMonitor()
        self.reporter = reporter


    def prepare(self):
        # The batch size will propagate from the input layer throughout the network.  This must be
        # done prior to network.prepare().
        self.network.input_layer.set_batch_size(self.batch_size)

        # Construct computation graphs, optimize, and compile.
        self.network.prepare()


    def train(self):
        batches = len(self.training_set)/self.batch_size
        for i in range(0, len(self.training_set), self.batch_size):
            x, z = zip(*self.training_set[i:i+self.batch_size])
            y, c = self.network.train(np.vstack(x), np.array(z))
            self.monitor.collect_statistics()
            logging.debug('actual: %5.2f; expected: %s' % (y[0,0], z[0,0]))
            if self.reporter:
                self.reporter.update()


class DataFeed(object):
    def __init__(self, monitor, layer, label):
        self.monitor = monitor
        self.layer = layer
        self.label = label

    def get_latest(self):
        return self.monitor.get_statistics(self.layer, self.label)[-1]


class DataSink(object):
    def put_result(self, result):
        pass


class FileSystemDataSink(DataSink):
    def __init__(self, root_dir, sub_dir, base_name, extension):
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
        if os.path.exists(self.link_path):
            os.remove(self.link_path)
        os.symlink(self.get_result_destination(), self.link_path)

    def get_result_destination(self):
        filename = '%s.%s.%s' % (self.base_name, self.count, self.extension)
        return os.path.join(self.files_dir, filename)



class FileSystemImageSink(FileSystemDataSink):
    def put_result(self, result):
        cv2.imwrite(result, self.get_result_destination())
        self.update_current_link()
        self.count += 1


class LoggingDataSink(DataSink):
    def __init__(self, format='%s', log_level='debug'):
        self.format = format
        self.log_level = log_level

    def put_result(self, result):
        getattr(logging, self.log_level)(self.format % result)


class DataReporter(object):
    def __init__(self, feeds=dict(), sink=None):
        self.feeds = dict(feeds)

        self.sink = sink
        if sink is None:
            self.sink = DataSink()


    def update(self):
        data = self.fetch()
        result = self.render(data)
        self.emit(result)

    def fetch(self):
        data = dict()
        for label, feed in self.feeds.items():
            data[label] = feed.get_latest()

        return data
    
    def render(self, data):
        return data

    def emit(self, result):
        self.sink.put_result(result)

        
class ImageReporter(DataReporter):
    def get_image(self):
        return self.image


class HistogramReporter(ImageReporter):
    def init(self, bin_count=10, hist_range=None, *args, **kwargs):
        self.bin_count = bin_count
        self.hist_range = hist_range
        super(HistogramReporter, self).__init__(*args, **kwargs)

    def render(self, data):
        plt.hist(data, bins=self.bin_count, range=self.hist_range)
        return self


class NetworkMonitor(object):
    def __init__(self):
        self.reset()

    def add_subject(self, layer, label=None, labels=None):
        if label is not None:
            labels = [label]

        if labels is None:
            labels = Layer.valid_tap_labels

        taps = self.taps.setdefault(layer, dict())
        stats = self.stats.setdefault(layer, dict())
            
        for label in labels:
            taps[label] = layer.add_tap(label)
            stats[label] = []

        return self


    def clear_statistics(self):
        self.stats = dict()
        return self

    def collect_statistics(self):
        if not self.paused:
            for layer, labelled_taps in self.taps.items():
                for label, tap in labelled_taps.items():
                    self.stats[layer][label].append(tap())

        return self


    def get_statistics(self, layer, label=None):
        if label is not None:
            return self.stats[layer][label]

        return self.stats[layer]


    def reset(self):
        self.taps = dict()
        self.paused = False
        self.clear_statistics()
        return self

    def pause(self):
        self.paused = True
        return self

    def resume(self):
        self.paused = False
        return self
        

class TrainingDataSet(object):
    pass

class CirclesTrainingDataSet(TrainingDataSet):
    def __init__(self, image_size=16, radius=3, noise=0.1):
        self.image_size = image_size
        self.radius = radius
        self.noise = noise

    def generate(self, N=1):
        result = []
        for i in range(N):
            color = random.uniform(0, 255)
            background = random.uniform(0, 255)
            img = background * np.ones((self.image_size, self.image_size))

            x = random.randrange(0, self.image_size)
            y = random.randrange(0, self.image_size)
            cv2.circle(img, (x, y), self.radius, color, -1)

            img += np.random.normal(0, abs(color - background)*self.noise, (self.image_size, self.image_size))
            m = img.min()
            if m < 0:
                img -= m

            M = img.max()
            if M > 255:
                img *= 255/M

            result.append((img.reshape(1, -1).astype(np.single), np.array([[x], [y]], dtype=np.single)))
        return result

class BarsAndStripesTrainingDataSet(TrainingDataSet):
    def __init__(self, image_size=4):
        self.image_size = image_size

    def generate(self, N=1):
        result = []
        for i in range(N):
            img = np.zeros((self.image_size, self.image_size))
            x = random.randint(0, 2*self.image_size-1)
            if x >= self.image_size:
                img[:, x - self.image_size] = 1
            else:
                img[x, :] = 1
                
    
            result.append((img.reshape(1, -1).astype(np.single), np.uint32(x)))
        return result

        
def test():

    layers = [InputLayer(16),
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), SimpleBackprop(), size=16),
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), SimpleBackprop(), size=8),
              OutputLayer(objective=ClassifyInput(), size=1)]

    N = Network(layers=layers)

    k = BarsAndStripesTrainingDataSet()

    training_set = k.generate(50)
    validation_set = k.generate(10)

    M = NetworkMonitor()
    M.add_subject(layers[1])
    M.add_subject(layers[2])

    c1 = layers[2]
    F_weights = DataFeed(monitor=M, layer=c1, label='weights')
    F_weight_change = DataFeed(monitor=M, layer=c1, label='weight_change')
    F_gradient = DataFeed(monitor=M, layer=c1, label='weight_gradient')
    F_cost = DataFeed(monitor=M, layer=c1, label='cost')
    F_stimulus = DataFeed(monitor=M, layer=c1, label='stimulus')
    L = LoggingDataSink()
    R = DataReporter(feeds={ #'weights': F_weights,
                             #'weight_change': F_weight_change,
                             'gradient': F_gradient,
                             'cost': F_cost,
                             'stimulus': F_stimulus,
                             },  sink=L)

    T = NetworkTrainer(N, training_set, validation_set, batch_size=10, monitor=M, reporter=R)

    T.prepare()
    T.train()

    return T
