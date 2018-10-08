import autograd.numpy as np
from .NAR_GP import NAR_GP

class NAR_Bagging:
    def __init__(self, num_models, dataset, bfgs_iter=100, debug=True):
        self.num_models = num_models
        self.models = []
        for i in range(self.num_models):
            self.models.append(NAR_GP(dataset, bfgs_iter=bfgs_iter, debug=debug))

    def train(self, scale=1.0):
        for i in range(self.num_models):
            self.models[i].train(scale=scale)

    def predict(self, test_x):
        num_test = test_x.shape[1]
        py = np.zeros((num_test))
        ps2 = np.zeros((num_test, num_test))
        for i in range(self.num_models):
            tmp_py, tmp_ps2 = self.models[i].predict(test_x)
            py = py + tmp_py/self.num_models
            ps2 = ps2 + (tmp_ps2 + np.square(tmp_py))/self.num_models
        ps2 = ps2 - np.square(py)
        return py, ps2

        
