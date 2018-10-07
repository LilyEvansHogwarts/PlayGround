import autograd.numpy as np
from autograd import grad
from .GP import GP
from .NN_GP import NN_GP
from .NN_scale_GP import NN_scale_GP
from .Multifidelity_GP import Multifidelity_GP

class Bagging:
    def __init__(self, name, num_models, dataset, bfgs_iter=100, debug=False, layer_sizes=[], activations=[], l1=0, l2=0):
        self.name = name
        self.num_models = num_models
        self.dataset = dataset
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = np.copy(activations)
        self.l1 = l1
        self.l2 = l2
        self.standardization()
        self.construct_model()

    def standardization(self):
        print('Bagging, start standardization...')
        if self.dataset.has_key('train_x'):
            self.in_mean = self.dataset['train_x'].mean(axis=1)
            self.in_std = self.dataset['train_x'].std(axis=1)
            self.dataset['train_x'] = ((self.dataset['train_x'].T - self.in_mean)/self.in_std).T
            self.out_mean = self.dataset['train_y'].mean()
            self.out_std = self.dataset['train_y'].std()
            self.dataset['train_y'] = (self.dataset['train_y'] - self.out_mean)/self.out_std
        else:
            ## standardize low_x and high_x
            self.in_mean = np.concatenate((self.dataset['low_x'].T, self.dataset['high_x'].T)).mean(axis=0)
            self.in_std = np.concatenate((self.dataset['low_x'].T, self.dataset['high_x'].T)).std(axis=0)
            self.dataset['low_x'] = ((self.dataset['low_x'].T - self.in_mean)/self.in_std).T
            self.dataset['high_x'] = ((self.dataset['high_x'].T - self.in_mean)/self.in_std).T
            ## standardize low_y and high_y
            self.out_low_mean = self.dataset['low_y'].mean()
            self.out_low_std = self.dataset['low_y'].std()
            self.dataset['low_y'] = (self.dataset['low_y'] - self.out_low_mean)/self.out_low_std
            self.out_mean = self.dataset['high_y'].mean()
            self.out_std = self.dataset['high_y'].std()
            self.dataset['high_y'] = (self.dataset['high_y'] - self.out_mean)/self.out_std
        print('Bagging, finish standardization')

    def construct_model(self):
        print('Bagging, start construct model...')
        if self.name == 'GP':
            self.models = [GP(self.dataset, bfgs_iter=self.bfgs_iter, debug=self.debug) for i in range(self.num_models)]
        elif self.name == 'Multifidelity_GP':
            self.models = [Multifidelity_GP(self.dataset, bfgs_iter=self.bfgs_iter, debug=self.debug) for i in range(self.num_models)]
        elif self.name == 'NAR_GP':
            self.models = [NAR_GP(self.dataset, bfgs_iter=self.bfgs_iter, debug=self.debug) for i in range(self.num_models)]
        elif self.name == 'NN_GP':
            self.models = [NN_GP(self.dataset, layer_sizes=self.layer_sizes, activations=self.activations, l1=self.l1, l2=self.l2, bfgs_iter=self.bfgs_iter, debug=self.debug) for i in range(self.num_models)]
        elif self.name == 'NN_scale_GP':
            self.models = [NN_scale_GP(self.dataset, layer_sizes=self.layer_sizes, activations=self.activations, l1=self.l1, l2=self.l2, bfgs_iter=self.bfgs_iter, debug=self.debug) for i in range(self.num_models)]
        else:
            print('There is no such gaussian process models as', self.name)
            sys.exit(1)
        print('Bagging, finish construct models')

    def train(self, scale=1.0):
        print('Bagging, start training the model')
        for i in range(self.num_models):
            self.models[i].train(scale=scale)
        print('Bagging, finish training the model')

    def predict(self, test_x):
        test_x = ((test_x.T - self.in_mean)/self.in_std).T
        py = np.zeros((test_x.shape[1]))
        ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(self.num_models):
            tmp_py, tmp_ps2 = self.models[i].predict(test_x)
            if self.debug:
                print('tmp_py',tmp_py.shape)
                print('tmp_ps2',tmp_ps2.shape)
            py += tmp_py/self.num_models
            ps2 += (tmp_ps2 + np.square(tmp_py))/self.num_models
            if self.debug:
                print('py',py.shape)
                print('ps2',ps2.shape)
        ps2 -= np.square(py)
        py = self.out_mean + py * self.out_std
        ps2 = ps2 * (self.out_std**2)
        return py, ps2






