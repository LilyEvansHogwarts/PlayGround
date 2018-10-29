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

    def wEI(self, x):
        x = x.reshape(self.dim, -1)
        py, ps2 = self.predict(x)
        ps = np.sqrt(ps2) + 0.000001
        EI = np.zeros((x.shape[1]))
        if self.best_constr[1] <= 0:
            tmp = -(py[0] - self.best_y[1,0])/ps[0]
            idx = (tmp > -6)
            EI[idx] = ps[0,idx]*(tmp[idx]*cdf(tmp[idx])+pdf(tmp[idx]))
            EI[idx] = np.log(np.maximum(EI[idx], 0.000001))
            idx = (tmp <= -6)
            tmp[idx] = tmp[idx]**2
            EI[idx] = np.log(ps[0,idx]) - tmp[idx]/2 - np.log(tmp[idx]-1)
        PI = np.zeros((x.shape[1]))
        for i in range(1,self.outdim):
            PI = PI + logphi_vector(-py[i]/ps[i])
        return EI+PI

    def predict(self, test_x):
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            py[i], ps2[i] = self.models[i].predict_for_wEI(test_x)
        return py, ps2

    def predict_low(self, test_x):
        num_test = test_x.shape[1]
        py = np.zeros((self.outdim, num_test))
        ps2 = np.zeros((self.outdim, num_test))
        for i in range(self.outdim):
            py[i], ps2[i] = self.models[i].predict_low(test_x)
        return py, ps2
