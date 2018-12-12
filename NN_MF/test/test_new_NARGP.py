import sys
sys.path.append('..')

import autograd.numpy as np
from print_out import *
from src.activations import *
from src.new_NARGP import new_NARGP
from src.Bagging import Bagging
from src.NNGP import NNGP

def branin_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    tmp1 = -1.275*np.square(x[0]/np.pi) + 5*x[0]/np.pi + x[1] - 6
    tmp2 = (10 - 5/(4*np.pi))*np.cos(x[0])
    ret = tmp1*tmp1 + tmp2 + 10
    return ret.reshape((1,-1))

def branin_low(x, bounds):
    tmp = (x.T - np.array([1.0/30, 11.0/30])).T
    tmp1 = branin_high(tmp, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = 10*np.sqrt(tmp1) + 2*(x[0]-0.5) - 3*(x[1]-1) - 1
    return ret.reshape((1,-1))

def get_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    x = np.random.uniform(-0.5, 0.5, (dim, num[0]))
    dataset = {}
    dataset['low_x'] = x
    dataset['high_x'] = x[:,:num[1]]
    dataset['low_y'] = funct[0](dataset['low_x'], bounds)
    dataset['high_y'] = funct[1](dataset['high_x'], bounds)
    return dataset

bounds = np.array([[-5,10],[0,15]])
dataset = get_dataset([branin_low, branin_high], np.array([300, 100]), bounds)

test_x = np.random.uniform(-0.5, 0.5, (bounds.shape[1], 20))
test_y = branin_high(test_x, bounds)

layer_sizes = np.array([20]*3)
activations = [relu]*3

model = new_NARGP(5, dataset, layer_sizes, activations, bfgs_iter=100)
model.train(scale=0.2)
py, ps2 = model.predict(test_x)
print_out(test_y, py, ps2)

'''
data = {}
data['train_x'] = dataset['low_x']
data['train_y'] = dataset['low_y']

model1 = Bagging(NNGP, 5, data, layer_sizes, activations, bfgs_iter=100)
model1.train(scale=0.2)
mu, _ = model1.predict(dataset['high_x'])


data['train_x'] = np.concatenate((dataset['high_x'], mu.reshape((1,-1))))
data['train_y'] = dataset['high_y']
model = GP(data, layer_sizes, activations, bfgs_iter=100, debug=False)
model.train(scale=0.2)

mu, v = model1.predict(test_x)
nsamples = 100
Z = np.random.multivariate_normal(mu, v, nsamples)
pys = np.zeros((nsamples, test_x.shape[1]))
ps2 = np.zeros(test_x.shape[1])
for i in range(nsamples):
    v = np.concatenate((test_x, Z[i].reshape((1,-1))))
    pys[i], tmp = model.predict(v)
    ps2 += np.diag(tmp)/nsamples
py = pys.mean(axis=0)
ps2 += pys.var(axis=0)

print_out(test_y, py, ps2)
'''
