from simanneal import Annealer
import numpy as np
import sys
import toml
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import *
import multiprocessing
from get_dataset import *


class fitting(Annealer):
    def __init__(self, x0, model):
        self.model = model
        self.state = np.copy(x0.reshape(-1))

    def move(self):
        self.state = self.state + 0.1*np.random.uniform(-0.5,0.5,(self.state.size))
        self.state = np.maximum(-0.5, np.minimum(0.5, self.state))

    def energy(self):
        x = self.state.reshape(self.model.dim,int(self.state.size/self.model.dim))
        EI = np.zeros((x.shape[1]))
        if self.model.best_constr[1] <= 0:
            _, _, py, ps2 = self.model.models[0].predict(x)
            ps = np.sqrt(ps2)
            tmp = -(py - self.model.best_y[1,0])/ps
            idx = (tmp > -40)
            EI[idx] = ps[idx]*(tmp[idx]*cdf(tmp[idx])+pdf(tmp[idx]))
            idx = (tmp <= -40)
            tmp2 = tmp[idx]**2
            EI[idx] = np.log(ps[idx]) - tmp2/2 - np.log(tmp2-1)
        PI = np.zeros((x.shape[1]))
        for i in range(1,self.model.outdim):
            _, _, py, ps2 = self.model.models[i].predict(x)
            ps = np.sqrt(ps2)
            PI = logphi(-py/ps) + PI
        loss = -EI-PI
        return loss.min()


argv = sys.argv[1:]
conf = toml.load(argv[0])

funct = get_funct(conf['funct'])
num = conf['num']
bounds = np.array(conf['bounds'])
scale = conf['scale']
bfgs_iter = conf['bfgs_iter']
num_points = conf['num_points']
iteration = conf['iteration']
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

    p = 10
    def task(x0):
        x0 = fit(x0, model)
        ff = fitting(x0, model)
        ff.copy_strategy = 'slice'
        ff.steps = 30
        ff.updates = 30
        x0, wEI_tmp = ff.anneal()
        x0 = x0.reshape(model.dim,int(x0.size/model.dim))
        wEI_tmp = model.wEI(x0)
        return x0, wEI_tmp

    pool = multiprocessing.Pool(processes=5)
    x0_list = []
    for j in range(int(K/p)):
        x0_list.append(model.rand_x(p))
    results = pool.map(task, x0_list)
    pool.close()
    pool.join()

    new_x = results[0][0]
    wEI_tmp = results[1][1]
    for j in range(1, int(K/p)):
        new_x = np.concatenate((new_x.T, results[j][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp, results[j][1].T)).T

    py, ps2 = model.predict_low(new_x)
    ps2 = ps2.sum(axis=0)
    idx = np.argsort(ps2)[-num_points:]
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





















