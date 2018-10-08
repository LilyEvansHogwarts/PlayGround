from src.NAR_GP import NAR_GP
import autograd.numpy as np
import matplotlib.pyplot as plt

def high(x):
    return (6.0*x-2.0)**2 * np.sin(12.0*x-4.0)

def low(x):
    return 0.5*high(x) + 10.0*(x-0.5) - 5.0
num_low = 15
num_high = 8
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

dataset = {}
dataset['low_x'] = low_x
dataset['low_y'] = low_y
dataset['high_x'] = high_x
dataset['high_y'] = high_y

model = NAR_GP(dataset, bfgs_iter=100, debug=True)
model.train(scale=0.4)
py, ps2 = model.predict(test_x)
ps2 = np.diag(ps2)
print('py',py)
print('true',test_y)
print('ps2',ps2)
delta = test_y - py
print('delta',delta)
print('MSE',np.dot(delta, delta.T))



plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
plt.plot(test_x[0], py, 'b-', label='prediction', linewidth=1)
plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
plt.fill_between(test_x[0], py-3*ps2, py+3*ps2, facecolor='orange', alpha=0.5, label='three std band')
plt.legend(frameon=False)
plt.show()




