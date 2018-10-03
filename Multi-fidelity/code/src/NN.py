import autograd.numpy as np
from .activations import *

class NN:
    def __init__(self, layer_sizes, activations):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = np.copy(activations)

    def num_param(self, dim):
        xs = [dim]
        for i in range(self.num_layers):
            xs.append(self.layer_sizes[i])
        results = 0
        for i in range(self.num_layers):
            results += (1+xs[i])*xs[i+1]
        return results

    def w_nobias(self, w, dim):
        prev_size = dim
        start_idx = 0
        wnb = np.array([])
        for i in range(self.num_layers):
            current_size = self.layer_sizes[i]
            num_w_layer = (1+prev_size)*current_size
            w_layer = np.reshape(w[start_idx:start_idx+num_w_layer],(1+prev_size,current_size))[:prev_size]
            wnb = np.concatenate((wnb, w_layer.reshape(w_layer.size)))
            start_idx += num_w_layer
            prev_size = current_size
        return wnb

    def predict(self, w, x):
        dim, num_train = x.shape
        start_idx = 0
        prev_size = dim
        out = x
        bias = np.ones((1,num_train))
        for i in range(self.num_layers):
            current_size = self.layer_sizes[i]
            num_w_layer = (1+prev_size)*current_size
            w_layer = np.reshape(w[start_idx:start_idx+num_w_layer],(1+prev_size,current_size))
            out = np.concatenate((out,bias))
            out = self.activations[i](np.dot(w_layer.T, out))
            prev_size = current_size
            start_idx += num_w_layer
        return out


