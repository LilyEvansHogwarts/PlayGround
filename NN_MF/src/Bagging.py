import autograd.numpy as np

class Bagging:
    def __init__(self, model, num_model, dataset, layer_sizes, activations, l1=0, l2=0, bfgs_iter=100, debug=False):
        self.model = model
        self.num_model = num_model
        self.dataset = dataset
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.l1 = l1
        self.l2 = l2
        self.bfgs_iter = bfgs_iter
        self.debug = debug

    def train(self, scale=0.2):
        self.models = []
        for i in range(self.num_model):
            m = self.model(self.dataset, self.layer_sizes, self.activations, l1=self.l1, l2=self.l2, bfgs_iter=self.bfgs_iter, debug=self.debug)
            m.train(scale=scale)
            self.models.append(m)

    def predict(self, test_x):
        pys = np.zeros((self.num_model, test_x.shape[1]))
        ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(self.num_model):
            pys[i], tmp = self.models[i].predict(test_x)
            ps2 += tmp/self.num_model
        py = pys.mean(axis=0)
        ps2 = ps2 + pys.var(axis=0)*np.eye(test_x.shape[1])
        return py, ps2
