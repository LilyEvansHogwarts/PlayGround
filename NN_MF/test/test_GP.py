import sys
sys.path.append('..')

import autograd.numpy as np
from src.GP import GP
from print_out import print_out

def branin(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    return (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def get_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    train_x = np.random.uniform(-0.5, 0.5, (dim, num))
    train_y = funct(train_x, bounds)
    return train_x, train_y

bounds = np.array([[-5,10],[0,15]])
train_x, train_y = get_dataset(branin, 100, bounds)

test_x = np.random.uniform(-0.5, 0.5, (bounds.shape[0], 20))
test_y = branin(test_x, bounds)

model = GP(train_x, train_y)
model.train(scale=0.4)
py, ps2 = model.predict(test_x)

print_out(test_y, py, ps2)
