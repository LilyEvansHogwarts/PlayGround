import sys
import toml
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import *
import multiprocessing
from get_dataset import *
import autograd.numpy as np

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
scale = conf['scale']
bfgs_iter = conf['bfgs_iter']
iteration = conf['iteration']
num_points = conf['num_points']
K = conf['K']



dataset = init_dataset(funct, num, bounds)
for i in dataset.keys():
    print(i, dataset[i].shape)




for i in range(iteration):
    print('********************************************************************')
    print('iteration',i)
    model = NAR_BO(dataset, scale, bounds, bfgs_iter=bfgs_iter, debug=False)
    best_x = model.best_x[1].reshape(-1,1)
    best_y = model.best_y[1].reshape(-1,1)
    best_y = model.re_standard(best_y)
    print('best_x', best_x)
    print('best_y', best_y)

    p = 20
    def task(x0):
        x0 = fit(x0, model)
        x0 = fit_test(x0, model)
        wEI_tmp = model.wEI(x0)
        return x0, wEI_tmp

    
    pool = multiprocessing.Pool(processes=5)
    di = []
    for j in range(int(K/p)):
        di.append(model.rand_x(p))
    results = pool.map(task, di)
    pool.close()
    pool.join()

    new_x = results[0][0]
    wEI_tmp = results[0][1]
    for j in range(1, int(K/p)):
        new_x = np.concatenate((new_x.T, results[j][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp, results[j][1].T)).T
    
    '''
    py, ps2 = model.predict_low(new_x)
    ps2 = ps2.sum(axis=0)
    '''
    idx = np.argsort(wEI_tmp)[-num_points-1:-1]
    print('idx',idx)
    print('low_x',new_x[:,idx])
    print('low_y', funct[0](new_x[:,idx],bounds))
    dataset['low_x'] = np.concatenate((dataset['low_x'].T, new_x[:,idx].T)).T
    dataset['low_y'] = np.concatenate((dataset['low_y'].T, funct[0](new_x[:,idx], bounds).T)).T

    idx = np.argsort(wEI_tmp)[-1:]
    print('idx', idx)
    print('high_x',new_x[:,idx])
    print('high_y', funct[1](new_x[:,idx], bounds))
    dataset['high_x'] = np.concatenate((dataset['high_x'].T, new_x[:,idx].T)).T
    dataset['high_y'] = np.concatenate((dataset['high_y'].T, funct[1](new_x[:,idx], bounds).T)).T







