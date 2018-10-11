import autograd.numpy as np
import sys
import toml
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import *
import multiprocessing
from get_dataset import *

def stand_print(x, py, ps2, true):
    print('x', x)
    print('py',py)
    print('ps2', ps2)
    print('true',true)
    delta = true - py
    print('delta',delta)
    print('MSE',np.dot(delta,delta.T))

argv = sys.argv[1:]
conf = toml.load(argv[0])

funct = get_funct(conf['funct'])
num = conf['num']
bounds = np.array(conf['bounds'])
num_models = conf['num_models']
scale = conf['scale']
bfgs_iter = conf['bfgs_iter']

dataset = init_dataset(funct, num, bounds)
for i in dataset.keys():
    print(i, dataset[i].shape)

num_models = 1
iteration = 10
K = 5
for i in range(iteration):
    model = NAR_BO(num_models, dataset, scale, bounds, bfgs_iter=bfgs_iter, debug=False)
    best_x = model.best_x[1].reshape(-1,1)
    best_y = model.best_y[1].reshape(-1,1)
    best_y = model.re_standard(best_y)
    print('best_x', best_x)
    print('best_y', best_y)

    test_x = model.rand_x(n=50)
    py1, ps21, py, ps2 = model.predict(test_x)
    # stand_print(test_x, py, ps2, funct[1](test_x, bounds))
    x = test_x.flatten()
    new_x = fit_test(x, model)
    wEI_tmp = model.wEI(new_x)
    py1, ps21, py, ps2 = model.predict(new_x)
    # stand_print(new_x, py, ps2, funct[1](new_x, bounds))
    
    ps21 = ps21.sum(axis=0)
    idx = np.argsort(ps21)[-K:]
    print('idx',idx)
    print('low_x',new_x[:,idx])
    print('low_y', funct[0](new_x[:,idx],bounds))
    dataset['low_x'] = np.concatenate((dataset['low_x'].T, new_x[:,idx].T)).T
    dataset['low_y'] = np.concatenate((dataset['low_y'].T, funct[0](new_x[:,idx], bounds).T)).T
    idx = np.argsort(wEI_tmp)[-1:]
    print('idx',idx)
    print('high_x',new_x[:,idx])
    print('high_y', funct[1](new_x[:,idx], bounds))
    dataset['high_x'] = np.concatenate((dataset['high_x'].T, new_x[:,idx].T)).T
    dataset['high_y'] = np.concatenate((dataset['high_y'].T, funct[1](new_x[:,idx], bounds).T)).T



