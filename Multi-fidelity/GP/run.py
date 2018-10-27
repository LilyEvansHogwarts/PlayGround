import autograd.numpy as np
import sys
import toml
from src.activations import *
from src.BO import BO
from src.fit import *
import multiprocessing
from get_dataset import *
import pickle

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
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
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)


for i in range(iteration):
    print('********************************************************************')
    print('iteration',i)
    model = BO(dataset, scale, bounds, bfgs_iter, debug=False)
    best_x = model.best_x
    best_y = model.best_y
    # best_y = model.re_standard(best_y)
    print('best_x', best_x)
    print('best_y', best_y)

    

    p = np.minimum(int(K/5), 5)
    def task(x0):
        # for i in range(x0.shape[1]):
        #     x0[:, i] = fit_py(x0[:, i], model, name)
        x0 = fit_new_py(x0, model)
        x0 = fit(x0, model)
        wEI_tmp = model.wEI(x0)
        return x0, wEI_tmp

    pool = multiprocessing.Pool(processes=5)
    x0_list = []
    for i in range(int(K/p)):
        x0_list.append(model.rand_x(p))
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
    new_y = funct[1](new_x[:, idx], bounds)
    print('idx',idx)
    print('x',new_x[:,idx].T)
    print('y',new_y.T)
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, new_x[:,idx].T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, new_y.T)).T
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    























    
