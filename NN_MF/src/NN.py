import autograd.numpy as np
from .activations import *

class NN:
    def __init__(self, dim, layer_sizes, activations):
        self.dim = dim
        self.num_layers = len(layer_sizes)
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = activations
        self.num_param = self.calc_num_param()
        

    def calc_num_param(self):
        pre_size = self.dim
        result = 0
        for i in range(self.num_layers):
            result += (pre_size + 1) * self.layer_sizes[i]
            pre_size = self.layer_sizes[i]
        return result

    def predict(self, w, x):
        pre_size, num = x.shape
        bias = np.ones((1,num))
        start_idx = 0
        out = x
        for i in range(self.num_layers):
            num_layer = (pre_size + 1) * self.layer_sizes[i]
            w_layer = w[start_idx:start_idx+num_layer]
            w_layer = w_layer.reshape((self.layer_sizes[i],-1))
            out = np.concatenate((out, bias))
            out = self.activations[i](np.dot(w_layer, out))
            pre_size = self.layer_sizes[i]
            start_idx += num_layer
        return out

    def w_nobias(self, w):
        pre_size = self.dim
        start_idx = 0
        w_nobia = np.array([])
        for i in range(self.num_layers):
            num_layer = (pre_size + 1) * self.layer_sizes[i]
            w_layer = w[start_idx:start_idx+num_layer].reshape((self.layer_sizes[i], pre_size+1))
            w_layer = w_layer[:,:pre_size]
            w_nobia = np.concatenate((w_nobia, w_layer.reshape(-1)))
            pre_size = self.layer_sizes[i]
            start_idx += num_layer
        return w_nobia



