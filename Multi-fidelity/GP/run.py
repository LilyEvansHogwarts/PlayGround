import autograd.numpy as np
import sys
import toml
from src.activations import *
from src.BO import BO
from src.fit import *
import multiprocessing
from get_dataset import *


argv = sys.argv[1:]
conf = toml.load(argv[0])

funct = get_funct(conf['funct'])
num = conf['num']
bounds = np.array(conf['bounds'])
scale = conf['scale']
bfgs_iter = conf['bfgs_iter']
iteration = conf['iteration']
K = conf['K']

data = init_dataset(funct, num, bounds)
dataset = {}
dataset['train_x'] = data['high_x']
dataset['train_y'] = data['high_y']

for i in range(iteration):
    print('********************************************************************')
    print('iteration',i)
    model = BO(dataset, scale, bounds, bfgs_iter, debug=False)
    best_x = model.best_x
    best_y = model.best_y
    best_y = model.re_standard(best_y)
    print('best_x', best_x)
    print('best_y', best_y)

    test_x = model.rand_x(n=K)
    new_x = fit(test_x, model)
    wEI_tmp = model.wEI(new_x)

    idx = np.argsort(wEI_tmp)[-1:]
    print('idx',idx)
    print('x',new_x[:,idx])
    print('y',funct[1](new_x[:,idx], bounds))
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, new_x[:,idx].T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, funct[1](new_x[:,idx], bounds).T)).T

    























    
