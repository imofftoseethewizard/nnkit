
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

        

logging.root.setLevel(logging.DEBUG)
theano.config.compute_test_value = 'warn'

def test():

    layers = [InputLayer(16),
              NeuronLayer(CompleteDendrite(), LogisticSynapse(), SimpleBackprop(), size=16),
              NeuronLayer(CompleteDendrite(), Synapse(), SimpleBackprop(), size=8),
              OutputLayer(objective=ClassifyInput(), size=1)]

    N = Network(layers=layers)

    k = BarsAndStripesTrainingDataSet()

    training_set = k.generate(4000)
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
