from .Bagging import Bagging
import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import traceback
import sys
from .activations import *

class BO:
    def __init__(self, name, num_models, dataset, bfgs_iter=100, debug=False, scale=[], num_layers=[], layer_sizes=[], activations=[], l1=0, l2=0):
        self.name = name
        self.num_models = num_models
        self.dataset = dataset
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.num_layers = np.copy(num_layers)
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = np.copy(activations)
        self.l1 = l1
        self.l2 = l2
        self.scale = np.copy(scale)
        self.construct_model()
        self.best_constr = np.inf
        self.best_y = np.zeros(self.outdim)
        self.best_y[0] = np.inf
        # get best_y from the dataset
        if self.dataset.has_key('train_x'):
            self.get_best_y(self.dataset['train_x'], self.dataset['train_y'])
        else:
            self.get_best_y(self.dataset['high_x'], self.dataset['high_y'])

    def construct_model(self):
        # get dim and outdim
        if self.name == 'GP' or self.name == 'NN_GP' or self.name == 'NN_scale_GP':
            self.dim = self.dataset['train_x'].shape[0]
            self.outdim = self.dataset['train_y'].shape[0]
        else:
            self.dim = self.dataset['low_x'].shape[0]
            self.outdim = self.dataset['low_y'].shape[0]
        # construct gaussian process model for each output index
        self.models = []
        for i in range(self.outdim):
            if self.name == 'NN_GP' or self.name == 'NN_scale_GP':
                layer_sizes = [self.layer_sizes[i]]*self.num_layers[i]
                act = [get_act_f(self.activations[i])]*self.num_layers[i]
                model = Bagging(self.name, self.num_models, self.dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug, layer_sizes=layer_sizes, activations=act, l1=self.l1[i], l2=self.l2[i])
                model.train(scale=self.scale[i])
            else:
                model = Bagging(self.name, self.num_models, self.dataset, bfgs_iter=self.bfgs_iter[i], debug=self.debug)
                model.train(scale=self.scale[i])
            self.models.append(model)

    def get_best_y(self, x, y):
        for i in range(y.shape[1]):
            constr = np.maximum(y[1:,i],0).sum()
            if self.best_constr > 0 and constr < self.best_constr:
                self.best_constr = constr
                self.best_x = x[:,i]
                self.best_y = y[:,i]
            elif self.best_constr <= 0 and constr <= 0 and y[0,i] < self.best_y[0]:
                self.best_constr = constr
                self.best_x = x[:,i]
                self.best_y = y[:,i]
    
    def predict(self, x):
        pys = np.zeros((x.shape[1], self.outdim))
        ps2s = np.zeros((x.shape[1], self.outdim))
        for i in range(self.outdim):
            py, ps2 = self.models[i].predict(x)
            pys[:,i] = py[:,0]
            ps2s[:,i] = np.diag(ps2)
        return pys, ps2s

    def wEI(self, x):
        x = x.reshape(self.dim, int(x.size/self.dim))
        EI = 1.0
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            tmp = (self.best_y[0] - py)/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = 1.0
        for i in range(1, self.outdim):
            py, ps2 = self.models[i].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            PI = PI*cdf(-py/ps)
        return EI*PI

    def fit(self, x):
        x0 = np.copy(x)
        self.x = np.copy(x)
        self.loss = np.inf

        def loss(x):
            loss = -self.wEI(x)
            if loss < self.loss:
                self.loss = loss
                self.x = np.copy(x)
            return loss

        gloss = grad(x)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=100, iprint=self.debug)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=10, iprint=self.debug)
            except:
                print('Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        if(np.isnan(self.loss) or np.isinf(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        return self.x


            

