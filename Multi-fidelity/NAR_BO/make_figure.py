import autograd.numpy as np
import sys
import toml
from src.GP import GP
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import *
import multiprocessing
from get_dataset import *
import pickle
import matplotlib.pyplot as plt
import GPy

def high(x, bounds):
    tmp = low(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    # return 0.1*(5.0*x - 1.0)**2 * np.sin(6.0*x-4.0)
    return (x - np.sqrt(2)) * tmp**2

def low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    # return 0.5*tmp + 0.2*(5.0*(x-0.5) + 2)
    return np.sin(8*np.pi*x)

def EI(model, x):
    py, ps2 = model.models[0].predict_for_wEI(x)
    ps = np.sqrt(ps2)
    tmp = -(py - model.best_y[1,0])/ps
    return ps*(tmp*cdf(tmp)+pdf(tmp))

def make_figure(model, dataset, test_x, test_y):
    py1, ps21 = model.models[0].predict_low(test_x)
    ps21 = np.sqrt(ps21)
    py2, ps22 = model.models[0].predict_for_wEI(test_x)
    ps22 = np.sqrt(ps22)
    low_x = dataset['low_x']
    low_y = dataset['low_y']
    high_x = dataset['high_x']
    high_y = dataset['high_y']

    plt.figure()
    plt.subplot(211)
    plt.plot(high_x[0], high_y[0], 'r*', markersize=4, label='high-fidelity data')
    plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
    plt.plot(test_x[0], py2, 'b--', label='prediction', linewidth=1)
    plt.fill_between(test_x[0], py2-2*ps22, py2+2*ps22, facecolor='lightgray', alpha=0.5, label='three std band')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('y', {'size':12})

    plt.subplot(212)
    plt.plot(test_x[0], EI(model, test_x))
    plt.yticks(fontsize=8)
    plt.ylabel('EI', {'size':12})
    plt.xticks(fontsize=8)
    plt.xlabel('x', {'size':12})
    plt.show()
    
    
    plt.figure()
    plt.subplot(211)
    plt.plot(low_x[0], low_y[0], 'ms', markersize=3, label='low-fidelity data')
    plt.plot(test_x[0], low(test_x[0], bounds), 'm--', label='exact', linewidth=1)
    plt.plot(high_x[0], high_y[0], 'r*', markersize=3, label='high-fidelity data')
    plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
    plt.plot(test_x[0], py2, 'b--', label='prediction', linewidth=1)
    plt.fill_between(test_x[0], py2-3*ps22, py2+3*ps22, facecolor='lightgray', alpha=0.5, label='three std band')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('y', {'size':12})

    plt.subplot(212)
    X = dataset['high_x'].T
    Y = dataset['high_y'].T
    kernel = GPy.kern.RBF(input_dim=1)
    m = GPy.models.GPRegression(X,Y,kernel)
    m.kern.variance = np.var(Y)
    m.kern.lengthscale = np.std(X)
    m.likelihood.variance = 0.01 * np.var(Y)
    m.optimize()
    y1, y2 = m.predict(test_x.T, full_cov=True)
    py = y1[:,0]
    ps = np.sqrt(np.diag(y2))
    plt.plot(high_x[0], high_y[0], 'r*', markersize=3, label='high-fidelity data')
    plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
    plt.plot(test_x[0], py, 'b--', label='prediction', linewidth=1)
    plt.fill_between(test_x[0], py-3*ps, py+3*ps, facecolor='lightgray', alpha=0.5, label='three std band')
    plt.yticks(fontsize=8)
    plt.ylabel('y', {'size':12})
    plt.xticks(fontsize=8)
    plt.xlabel('x', {'size':12})
    plt.show()



    '''
    plt.subplot(212)
    data = {}
    data['train_x'] = dataset['high_x']
    data['train_y'] = dataset['high_y']
    model = GP(data, bfgs_iter=2000, debug=False)
    model.train(scale=0.1)
    py, ps2 = model.predict(test_x)
    print(ps2)
    ps2 = np.sqrt(ps2)
    plt.plot(high_x[0], high_y[0], 'r*', markersize=3, label='high-fidelity data')
    plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
    plt.plot(test_x[0], py, 'b--', label='prediction', linewidth=1)
    plt.fill_between(test_x[0], py-3*ps2, py+3*ps2, facecolor='lightgray', alpha=0.5, label='three std band')
    plt.yticks(fontsize=8)
    plt.ylabel('y', {'size':12})
    plt.xticks(fontsize=8)
    plt.xlabel('x', {'size':12})
    plt.show()
    '''

def stand_print(x, py, ps2, true):
    print('x', x)
    print('py',py)
    print('ps2', ps2)
    print('true',true)
    delta = true - py
    print('delta',delta)
    print('MSE',np.dot(delta,delta.T))

num = np.array([50,10])
bounds = np.array([[0,1]])
scale = np.array([0.1])
bfgs_iter = np.array([2000])
iteration = 0
K = 100
gamma = 0.01

test_x = np.random.rand(1, 1000) - 0.5
test_x.sort()
test_y = high(test_x, bounds)

funct = [low, high]

dataset = init_dataset(funct, num, bounds)

dataset['high_x'][0,0] = -0.32
dataset['high_x'][0,1] = 0.41
dataset['high_x'][0,2] = 0.25
# dataset['high_x'][0,3] = -0.05
dataset['high_y'] = high(dataset['high_x'], bounds)

dataset['low_x'][0,0] = -0.45
dataset['low_x'][0,1] = 0.42
dataset['low_y'] = low(dataset['low_x'], bounds)

with open('make_figure.pickle', 'rb') as f:
    dataset = pickle.load(f)
model = NAR_BO(dataset, gamma, scale, bounds, bfgs_iter=bfgs_iter, debug=False)
make_figure(model, dataset, test_x, test_y)
'''
data = {}
data['train_x'] = dataset['high_x']
data['train_y'] = dataset['high_y']
model = GP(data, bfgs_iter=2000, debug=True)
model.train(scale=0.0001)
py, ps2 = model.predict(test_x, is_diag=1)
ps2 = np.sqrt(ps2)
print(ps2)

plt.figure()
plt.plot(test_x[0], py-3*ps2)
plt.plot(test_x[0], py+3*ps2)
plt.plot(test_x[0], py)
plt.plot(dataset['high_x'][0], dataset['high_y'][0], 'r*')
plt.show()
'''
