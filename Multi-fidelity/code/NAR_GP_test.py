from src.NAR_GP import NAR_GP
import autograd.numpy as np
import matplotlib.pyplot as plt

def high(x):
    return (6.0*x-2.0)**2 * np.sin(12.0*x-4.0)

def low(x):
    return 0.5*high(x) + 10.0*(x-0.5) - 5.0
num_low = 8
num_high = 4
num_test = 200

low_x = np.arange(0,num_low)/(1.0*num_low)
low_x = np.concatenate((low_x, np.array([1.0])))
low_x = low_x.reshape(1,num_low+1)
low_y = low(low_x)

high_x = np.arange(0,num_high)/(1.0*num_high)
high_x = np.concatenate((high_x, np.array([1.0])))
high_x = high_x.reshape(1,num_high+1)
high_y = high(high_x)

test_x = np.arange(0,num_test)/(1.0*num_test)
test_x = np.concatenate((test_x, np.array([1.0])))
test_x = test_x.reshape(1,num_test+1)
test_y = high(test_x)

dataset = {}
dataset['low_x'] = low_x
dataset['low_y'] = low_y
dataset['high_x'] = high_x
dataset['high_y'] = high_y

model = NAR_GP(dataset, bfgs_iter=100, debug=True)
model.train(scale=0.4)
'''
plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
plt.plot(test_x[0], py[:,0], 'b-', label='prediction', linewidth=1)
plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
plt.fill_between(test_x[0], py[:,0]-3*np.diag(ps2), py[:,0]+3*np.diag(ps2), facecolor='orange', alpha=0.5, label='three std band')
plt.legend(frameon=False)
plt.show()
'''



