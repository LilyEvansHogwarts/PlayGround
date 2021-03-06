from src.Bagging import Bagging
import autograd.numpy as np
import matplotlib.pyplot as plt
from src.BO import BO

def high(x):
    return (6.0*x-2.0)**2 * np.sin(12.0*x-4.0)

def low(x):
    return 0.5*high(x) + 10.0*(x-0.5) - 5.0
num_low = 50
num_high = 10
num_test = 200

low_x = np.random.rand(1,num_low)
low_x.sort()
low_y = low(low_x)

high_x = np.random.rand(1,num_high)
high_x.sort()
high_y = high(high_x)

test_x = np.random.rand(1,num_test)
test_x.sort()
test_y = high(test_x)

print('low_x',low_x.shape)
print('low_y',low_y.shape)
print('high_x',high_x.shape)
print('high_y',high_y.shape)
print('test_x',test_x.shape)
print('test_y',test_y.shape)


dataset = {}
dataset['low_x'] = low_x
dataset['low_y'] = low_y
dataset['high_x'] = high_x
dataset['high_y'] = high_y


model = Bagging('Multifidelity_GP', 10, dataset, bfgs_iter=100, debug=False)
model.train(scale=0.4)
py, ps2 = model.predict(test_x)
print('py',py)
print('true',test_y)
print('ps2',np.diag(ps2))
delta = test_y - py
print('delta',delta)
print('MSE',np.dot(delta,delta.T))

'''
model = BO('Multifidelity_GP', 2, dataset, bfgs_iter=[100], debug=False, scale=[0.4])
py, ps2 = model.predict(test_x)
print('py',py.T)
print('true',test_y)
print('ps2',ps2.T)
delta = py.T - test_y
print('delta',delta)
print('MSE',np.dot(delta, delta.T))
'''
plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
plt.plot(test_x[0], py, 'b-', label='prediction', linewidth=1)
plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
plt.fill_between(test_x[0], py-3*np.diag(ps2), py+3*np.diag(ps2), facecolor='orange', alpha=0.5, label='three std band')
plt.legend(frameon=False)
plt.show()


