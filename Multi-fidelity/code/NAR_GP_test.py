from src.NAR_GP import NAR_GP
import autograd.numpy as np
import matplotlib.pyplot as plt

def high(x):
    return (6.0*x-2.0)**2 * np.sin(12.*x-4.0)

def low(x):
    return 0.5*high(x) + 10.0*(x-0.5) - 5.0
num_low = 50
num_high = 5
num_test = 200

low_x = np.arange(0,num_low)/(1.0*num_low)
low_x = np.concatenate((low_x, np.array([1.0])))
low_y = np.arange(0,num_low+1)
for i in range(num_low+1):
    low_y[i] = low(low_x[i])
low_x = low_x.reshape(1,num_low+1)
low_y = low_y.reshape(1,num_low+1)

high_x = np.arange(0,num_high)/(1.0*num_high)
high_x = np.concatenate((high_x, np.array([1.0])))
high_y = np.arange(0,num_high+1)
for i in range(num_high+1):
    high_y[i] = high(high_x[i])
high_x = high_x.reshape(1,num_high+1)
high_y = high_y.reshape(1,num_high+1)

test_x = np.arange(0,num_test)/(1.0*num_test)
test_x = np.concatenate((test_x, np.array([1.0])))
test_y = np.arange(0,num_test+1)
for i in range(num_test+1):
    test_y[i] = high(test_x[i])
test_x = test_x.reshape(1,num_test+1)
test_y = test_y.reshape(1,num_test+1)

model = NAR_GP(low_x, low_y, high_x, high_y, bfgs_iter=100, debug=False)
model.train()
py, ps2 = model.predict(test_x)
print('py',py.T)
print('true',test_y)
print('ps2',ps2)
delta = py[:,0] - test_y[0]
print('delta',delta)
print('MSE', np.dot(delta, delta))

plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
plt.plot(test_x[0], py[:,0], 'b-', label='prediction', linewidth=1)
plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
plt.fill_between(test_x[0], py[:,0]-3*np.diag(ps2), py[:,0]+3*np.diag(ps2), facecolor='orange', alpha=0.5, label='three std band')
plt.legend(frameon=False)
plt.show()




