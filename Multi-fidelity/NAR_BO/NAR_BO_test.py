from src.NAR_BO import NAR_BO
import autograd.numpy as np
import matplotlib.pyplot as plt

def high(x):
    return (6.0*x-2.0)**2 * np.sin(12.0*x-4.0)

def low(x):
    return 0.5*high(x) + 10.0*(x-0.5) - 5.0

num_low = 10
num_high = 3
num_test = 200

low_x = np.random.uniform(0,1,(1,num_low))
low_x.sort()
low_y = low(low_x)

high_x = np.random.uniform(0,1,(1,num_high))
high_x.sort()
high_y = high(high_x)

test_x = np.random.uniform(0,1,(1,num_test))
test_x.sort()
test_y = high(test_x)

dataset = {}
dataset['low_x'] = low_x
dataset['low_y'] = low_y
dataset['high_x'] = high_x
dataset['high_y'] = high_y

iteration = 20
K = 50
num_points = 3
for i in range(iteration):
    print('*************************************************************')
    print('iteration',i)
    model = NAR_BO(5, dataset, scale=[0.4], bfgs_iter=[100], debug=False)
    print('Finish building the model')

    py1, ps21, py, ps2 = model.predict(test_x)
    delta = low(test_x) - py1
    print('low MSE', np.dot(delta, delta.T))
    delta = test_y - py
    print('high MSE', np.dot(delta, delta.T))

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
    plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
    plt.plot(test_x[0], py1[0], 'b-', label='low prediction', linewidth=1)
    plt.plot(test_x[0], low(test_x[0]), 'r-', label='low exact', linewidth=1)
    plt.fill_between(test_x[0], py1[0]-3*ps21[0], py1[0]+3*ps21[0], facecolor='orange', alpha=0.5, label='low 3 std band')
    ax = plt.gca()
    ax.set_ylim([-20,20])
    # plt.savefig('./figures/low_%d.png' %(i), format='png', dpi=300)
    
    plt.subplot(1,2,2)
    plt.plot(low_x[0], low_y[0], 'bo', markersize=3, label='low-fidelity data')
    plt.plot(high_x[0], high_y[0], 'ms', markersize=5, label='high-fidelity data')
    plt.plot(test_x[0], py[0], 'b-', label='prediction', linewidth=1)
    plt.plot(test_x[0], test_y[0], 'r-', label='exact', linewidth=1)
    plt.fill_between(test_x[0], py[0]-3*ps2[0], py[0]+3*ps2[0], facecolor='orange', alpha=0.5, label='3 std band')
    ax = plt.gca()
    ax.set_ylim([-20,20])
    plt.savefig('./figures/BO_%d.png' %(i), format='png', dpi=300)

    x = np.random.uniform(0,1,(1,K))
    py1, ps21, py, ps2 = model.predict(x)
    idx = np.argsort(ps21[0])[-num_points:]
    low_x = np.concatenate((low_x.T, x[:,idx].T)).T
    tmp = model.EI(x, is_high=1)
    idx = np.argsort(tmp)[-1:]
    high_x = np.concatenate((high_x.T, x[:,idx].T)).T
    low_y = low(low_x)
    high_y = high(high_x)
    dataset['low_x'] = low_x
    dataset['low_y'] = low_y
    dataset['high_x'] = high_x
    dataset['high_y'] = high_y
    print('Finish updating dataset')



