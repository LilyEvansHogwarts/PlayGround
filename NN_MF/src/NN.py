import autograd.numpy as np

class NN:
    def __init__(self, layer_sizes, activations):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations

    def num_param(self, dim):
        xs = [dim]
        results = 0
        for l in self.layer_sizes:
            xs.append(l)
        for i in range(self.num_layers):
            results += (1+xs[i])*xs[i+1]
        return results

    def w_nobias(self, w, dim):
        prev_size = dim
        start_idx = 0
        wnb = np.array([])
        for i in range(self.num_layers):
            curr_size = self.layer_sizes[i]
            w_num_layer = (1+prev_size)*curr_size
            w_layer = w[start_idx:(start_idx+w_num_layer)].reshape(1+prev_size,curr_size)[:prev_size]
            wnb = np.concatenate((wnb, w_layer.reshape(-1)))
            start_idx += w_num_layer
            prev_size = curr_size
        return wnb

    def predict(self, w, x):
        dim, num_train = x.shape
        out = x
        start_idx = 0
        prev_size = dim
        bias = np.ones((1, num_train))
        for i in range(self.num_layers):
            curr_size = self.layer_sizes[i]
            w_num_layer = (1+prev_size)*curr_size
            w_layer = w[start_idx:(start_idx+w_num_layer)].reshape(1+prev_size,curr_size)
            out = np.concatenate((out, bias))
            out = self.activations[i](np.dot(w_layer.T, out))
            start_idx += w_num_layer
            prev_size = curr_size
        return out
