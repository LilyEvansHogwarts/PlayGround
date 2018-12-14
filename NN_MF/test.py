from src.NARGP import NARGP 
from src.scaled_NN_NARGP import scaled_NN_NARGP
from src.new_NARGP import new_NARGP
from src.activations import *
import autograd.numpy as np
from test.print_out import *


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
test_x = np.random.uniform(-0.5, 0.5, (bounds.shape[0], 800))
test_y = branin_high(test_x, bounds)

model1 = NARGP(dataset)
model1.train(scale=0.2)
py, ps2 = model1.predict(test_x)

print_out(test_y, py, ps2)

num_layer = 3
layer_sizes = np.array([100]*num_layer)
activations = [relu]*num_layer
model2 = scaled_NN_NARGP(5, dataset, layer_sizes, activations, l1=0, l2=0.0001, bfgs_iter=1000)
model2.train(scale=0.2)
py, ps2 = model2.predict(test_x)

print_out(test_y, py, ps2)
'''
model3 = new_NARGP(5, dataset, layer_sizes, activations, bfgs_iter=800)
model3.train(scale=0.2)
py, ps2 = model3.predict(test_x)

print_out(test_y, py, ps2)
'''
