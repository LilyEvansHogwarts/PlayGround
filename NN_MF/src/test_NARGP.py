import autograd.numpy as np
from NARGP import NARGP

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
    x = np.random.uniform(-0.5, 0.5, (dim, num.sum()))
    dataset = {}
    dataset['low_x'] = x[:,:num[0]]
    dataset['high_x'] = x[:,num[0]:]
    dataset['low_y'] = funct[0](dataset['low_x'], bounds)
    dataset['high_y'] = funct[1](dataset['high_x'], bounds)
    return dataset

bounds = np.array([[-5,10],[0,15]])
dataset = get_dataset([branin_low, branin_high], np.array([100, 30]), bounds)

model = NARGP(dataset, bfgs_iter=100)
model.train(scale=0.2)

test_x = np.random.uniform(-0.5, 0.5, (bounds.shape[0], 20))
test_y = branin_high(test_x, bounds)
print(test_y)

py, ps2 = model.predict(test_x)
print(py)
print(test_y - py)
print(np.sqrt(ps2))

