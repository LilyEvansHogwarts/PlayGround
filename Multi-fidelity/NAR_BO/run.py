import autograd.numpy as np
import sys
import toml
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import *
import multiprocessing
from get_dataset import *
import pickle

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

name = conf['funct']
funct = get_funct(name)
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
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)
'''
with open('dataset.pickle', 'rb') as f:
    data = pickle.load(f)
dataset = {}
dataset['low_x'] = data['low_x'][:, :20]
dataset['low_y'] = data['low_y'][:, :20]
dataset['high_x'] = data['high_x'][:, :10]
dataset['high_y'] = data['high_y'][:, :10]
'''

i = 0
while (dataset['high_y'].shape[1] - num[1]) <= iteration:
    print('********************************************************************')
    print('iteration',i)
    i = i+1
    for j in dataset.keys():
        print(j, dataset[j].shape)
    model = NAR_BO(dataset, scale, bounds, bfgs_iter=bfgs_iter, debug=False)
    best_x = model.best_x[1].reshape(-1,1)
    best_y = model.best_y[1].reshape(-1,1)
    print('best_x', best_x.T)
    print('best_y', best_y.T)

    
    p = 5
    def task(x0):
        x0 = fit(x0, model)
        for i in range(x0.shape[1]):
            x0[:, i] = fit_py(x0[:, i], model, name)
        x0 = fit_test(x0, model)
        wEI_tmp = model.wEI(x0)
        return x0, wEI_tmp

    pool = multiprocessing.Pool(processes=5)
    x0 = model.rand_x(K)
    x0_list = []
    for j in range(int(K/p)):
        x0_list.append(x0[:, p*j:p*(j+1)])
    results = pool.map(task, x0_list)
    pool.close()
    pool.join()

    new_x = results[0][0]
    wEI_tmp = results[0][1]
    for j in range(1, int(K/p)):
        new_x = np.concatenate((new_x.T, results[j][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp.T, results[j][1].T)).T
    # wEI_tmp = model.wEI(new_x)

    idx = np.argsort(wEI_tmp)[-1:]
    print('idx', idx)
    print('x', new_x[:, idx].T)
    py, ps2 = model.predict_low(new_x[:, idx])
    if (ps2.T > model.gamma).sum() > 0:
        new_y = funct[0](new_x[:, idx], bounds)
        print('low_y', new_y.T)
        dataset['low_x'] = np.concatenate((dataset['low_x'].T, new_x[:,idx].T)).T
        dataset['low_y'] = np.concatenate((dataset['low_y'].T, new_y.T)).T
    else:
        new_y = funct[1](new_x[:, idx], bounds)
        print('high_y', new_y.T)
        dataset['high_x'] = np.concatenate((dataset['high_x'].T, new_x[:,idx].T)).T
        dataset['high_y'] = np.concatenate((dataset['high_y'].T, new_y.T)).T
    
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)









