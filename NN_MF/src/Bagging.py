import autograd.numpy as np
from .NN_GP import NN_GP

class Bagging:
    def __init__(self, num_models, dataset, layer_sizes, activations, bfgs_iter, l1=0, l2=0, debug=True):
        self.num_models = num_models
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y'].reshape(-1)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bfgs_iter = bfgs_iter
        self.l1 = l1
        self.l2 = l2
        self.debug = debug
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y - self.mean)/self.std

    def train(self, scale=1.0):
        self.models = []
        for i in range(self.num_models):
            model = NN_GP(self.train_x, self.train_y, self.layer_sizes, self.activations, self.bfgs_iter, l1=self.l1, l2=self.l2, debug=self.debug)
            model.train(scale=scale)
            self.models.append(model)
        print('Bagging. Finish training process.')

    def predict(self, test_x, is_diag=1):
        num_test = test_x.shape[1]
        py = np.zeros((num_test))
        tmp = np.zeros((num_test))
        if is_diag:
            ps2 = np.zeros((num_test))
        else:
            ps2 = np.zeros((num_test, num_test))
        for i in range(self.num_models):
            tmp_py, tmp_ps2 = self.models[i].predict(test_x, is_diag=is_diag)
            py += tmp_py/self.num_models
            tmp += (tmp_py**2)/self.num_models
            ps2 += tmp_ps2/self.num_models
        if is_diag:
            ps2 = ps2 + tmp - py**2
        else:
            ps2 = ps2 + (tmp - py**2)*np.eye(num_test)
        py = py * self.std + self.mean
        ps2 = ps2 * (self.std**2)
        return py, ps2

