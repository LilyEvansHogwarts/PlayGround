import autograd.numpy as np
from autograd import grad
from .NAR_GP import NAR_GP

class NAR_BO:
    def __init__(self, num_models, dataset, gamma, scale, bounds, bfgs_iter, debug=True):
        self.dataset = {}
        self.dataset['low_x'] = np.copy(dataset['low_x'])
        self.dataset['low_y'] = np.copy(dataset['low_y'])
        self.dataset['high_x'] = np.copy(dataset['high_x'])
        self.dataset['high_y'] = np.copy(dataset['high_y'])
        self.gamma = self.dataset['high_y'].shape[0] * gamma * (self.dataset['low_y'].max(axis=1) - self.dataset['low_y'].min(axis=1))
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.scale = np.copy(scale)
        self.bounds = np.copy(bounds)
        self.dim = dataset['low_x'].shape[0]
        self.outdim = dataset['low_y'].shape[0]
        self.num_low = dataset['low_y'].shape[1]
        self.num_high = dataset['high_y'].shape[1]
        self.construct_model()

    def construct_model(self):
        dataset = {}
        dataset['low_x'] = self.dataset['low_x']
        dataset['high_x'] = self.dataset['high_x']
        self.models = []
        for i in range(self.outdim):
            dataset['low_y'] = self.dataset['low_y'][i]
            dataset['high_y'] = self.dataset['high_y'][i]
            model = NAR_GP(self.num_models, dataset, self.layer_sizes, self.activations, self.bfgs_iter, debug=self.debug)
            model.train(scale=self.scale[i])
            self.models.append(model)
        print('NAR_BO. Finish model construction')

    def rand_x(self, n=1):
        tmp = np.random.uniform(0,1,(n))
        idx = (tmp < 0.4)
        x = np.random.uniform(-0.5, 0.5, (self.dim, n))
        x[:,idx] = (0.05 * np.random.uniform(-0.5, 0.5, (self.dim, idx.sum())).T + self.best_x[1]).T
        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))

        idx = (tmp < 0.5) * (tmp > 0.4)
        x[:,idx] = (0.05 * np.random.uniform(-0.5, 0.5, (self.dim, idx.sum())).T + self.best_x[0]).T
        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))
        return x
