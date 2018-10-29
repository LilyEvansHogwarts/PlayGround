import autograd.numpy as np
from src.NAR_BO import NAR_BO
from src.activations import *
from src.NAR_GP import NAR_GP
from get_dataset import *
import sys
import toml
import pickle
import multiprocessing

argv = sys.argv[1:]
conf = toml.load(argv[0])

num_layers = conf['num_layers']
layer_sizes = conf['layer_sizes']
activations = conf['activations']
gamma = conf['gamma']
scale = conf['scale']
bounds = np.array(conf['bounds'])
bfgs_iter = conf['bfgs_iter']
l1 = conf['l1']
l2 = conf['l2']
num = conf['num']
name = conf['funct']
funct = get_funct(name)
num_models = conf['num_models']
iteration = conf['iteration']

dataset = init_dataset(funct, num, bounds)
with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

i = 0
while(dataset['high_y'].shape[1] - num[1]) <= iteration:
    print('**********************************************************')
    print('iteration',i)
    i = i+1
    for j in dataset.keys():
        print(j, dataset[j].shape)
    model = NAR_BO(num_models, dataset, num_layers, layer_sizes, activations, gamma, scale, bounds, bfgs_iter, l1=l1, l2=l2, debug=False)
    print('best_x', model.best_x[1])
    print('best_y', model.best_y[1])

    p = np.minimum(int(K/5), 5)
    def task(x0):
        x0 = fit_low(x0, model)
        x0 = fit_py(x0, model)
        x0 = fit_high(x0, model)
        wEI_tmp = model.wEI(x0)
        return x0, wEI_tmp

    pool = multiprocessing.Pool(processes=5)
    x0 = model.rand_x(K)
    x0_list = []
    for j in range(int(K/p)):
        x0_list.append(x0[:,p*j:p*(j+1)])
    results = pool.map(task, x0_list)
    pool.close()
    pool.join()

    new_x = results[0][0]
    wEI_tmp = results[0][1]
    for j in range(1, int(K/p)):
        new_x = np.concatenate((new_x.T, results[j][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp.T, results[j][1].T)).T

    idx = np.argsort(wEI_tmp)[-1:]
    print('idx',idx)
    print('x', new_x[:,idx].T)
    py, ps2 = model.predict_low(new_x[:,idx])
    if (ps2.T > model.gamma).sum() > 0:
        new_y = funct[0](new_x[:,idx], bounds)
        print('low_y', new_y.T)
        dataset['low_x'] = np.concatenate((dataset['low_x'].T, new_x[:,idx].T)).T
        dataset['low_y'] = np.concatenate((dataset['low_y'].T, new_y.T)).T
    else:
        new_y = funct[1](new_x[:,idx], bounds)
        print('high_y', new_y.T)
        dataset['high_x'] = np.concatenate((dataset['high_x'].T, new_x[:,idx].T)).T
        dataset['high_y'] = np.concatenate((dataset['high_y'].T, new_y.T)).T

    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)



