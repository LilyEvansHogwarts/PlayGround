import sys
import toml
from get_dataset import *
from src.Bagging_Constr_model import Bagging_Constr_model
import multiprocessing
from src.fit import *
import random

argv = sys.argv[1:]
conf = toml.load(argv[0])

l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']
num_layers = conf['num_layers']
layer_size = conf['layer_size']
act = conf['activation']
max_iter = conf['max_iter']
bounds = conf['bounds']
dim = conf['dim']
outdim = conf['outdim']
num_train = conf['num_train']
num_test = conf['num_test']
funct = conf['funct']
iteration = conf['iter']
K = conf['K']
num_models = conf['num_models']

main_f = get_main_f(funct)

dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

i = 0
while i < iteration:
    model = Bagging_Constr_model(num_models, main_f, dataset, dim, outdim, [[0,50]]*dim, scale, num_layers, layer_size, act, max_iter, l1=l1, l2=l2, debug=True)

    def task(tmp):
        x0 = model.rand_x()
        x0 = fit(x0, model)
        p = model.wEI(x0)
        return x0, p
    pool = multiprocessing.Pool(processes=5)
    results = pool.map(task, range(K))
    pool.close()
    pool.join()

    best_acq = -np.inf
    best_x = np.zeros((model.dim, 1))
    for j in range(K):
        p = results[j][1]
        x0 = results[j][0]
        if p > best_acq:
            best_acq = p
            best_x = x0.copy()
    py, ps2 = model.predict(best_x)
    best_y = main_f(get_eval(best_x,bounds)).T
    if (best_y == np.inf).sum() > 0:
        continue
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, best_x.T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, best_y)).T

    print('iteration', i)
    i = i+1
    print('best_acq', best_acq)
    print('best_constr', np.maximum(py[0, 1:], 0).sum())
    print('best_x', best_x.T)
    print('py', py)
    print('ps', np.sqrt(ps2))
    print('true', best_y)
    print()

    all_constr = model.best_constr
    all_loss = model.best_y
    all_x = model.best_x.reshape(model.dim, 1)
    all_y = model.best_out
    constr = np.maximum(best_y[0, 1:], 0).sum()
    if all_constr > 0 and constr < all_constr:
        all_constr = constr
        all_loss = best_y[0, 0]
        all_x = best_x.copy()
        all_y = best_y.copy()
    elif all_constr <= 0 and constr <= 0 and best_y[0, 0] < all_loss:
        all_constr = constr
        all_loss = best_y[0, 0]
        all_x = best_x.copy()
        all_y = best_y.copy()

    py, ps2 = model.predict(all_x)
    print('all_constr', all_constr)
    print('all_loss', all_loss)
    print('all_x', all_x.T)
    print('py', py)
    print('ps', np.sqrt(ps2))
    print('true', all_y)
    print('--------------------------------------------------------------------------------------------')

with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)
