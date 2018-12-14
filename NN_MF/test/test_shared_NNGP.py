import sys
sys.path.append('..')

import autograd.numpy as np
from src.shared_NNGP import shared_NNGP
from src.activations import *
from src.NN import NN
from print_out import *

enb = np.loadtxt('enb.txt')
print(enb.shape)

num_train = 700
dataset = {}
dataset['train_x'] = enb[:num_train,:8].T
dataset['train_y'] = enb[:num_train,8:].T

test_x = enb[num_train:,:8].T
test_y = enb[num_train:,8:].T


models = []

for i in range(5):
    shared_nn = NN(8, np.array([100]*2), [relu]*2)
    nns = [NN(100, np.array([100]), [relu]), NN(100, np.array([100]), [relu])]
    model = shared_NNGP(dataset, shared_nn, nns, bfgs_iter=1000, debug=False)
    model.train(scale=0.2)
    models.append(model)

pys = np.zeros((5, 2, test_x.shape[1]))
ps2s = np.zeros((5, 2, test_x.shape[1]))
for i in range(5):
    pys[i], ps2s[i] = models[i].predict(test_x)
ps2s = ps2s.mean(axis=0) + pys.var(axis=0)
pys = pys.mean(axis=0)

for i in range(2):
    print_out(test_y[i], pys[i], ps2s[i])
