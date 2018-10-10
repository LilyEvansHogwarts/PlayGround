import autograd.numpy as np

def branin_high(x):
    tmp1 = -1.275*np.square(x[0]/np.pi) + 5*x[0]/np.pi + x[1] - 6
    tmp2 = (10 - 5/(4*np.pi))*np.cos(x[0])
    ret = tmp1*tmp1 + tmp2 + 10
    return ret.reshape(1,-1)

def branin_low(x):
    ret = 10*np.sqrt(branin_high(x-2)) + 2*(x[0]-0.5) - 3*(x[1]-1) - 1
    return ret.reshape(1,-1)

def init_dataset(funct, num, bounds):
    dim = len(bounds)
    total = num[0]+num[1]
    x = np.zeros((dim, total))
    for i in range(dim):
        x[i] = np.random.uniform(bounds[i][0], bounds[i][1], total)
    dataset = {}
    dataset['low_x'] = x[:,:num[0]]
    dataset['high_x'] = x[:,num[0]:]
    dataset['low_y'] = funct[0](dataset['low_x'])
    dataset['high_y'] = funct[1](dataset['high_x'])
    return dataset

def get_test(funct, num, bounds):
    dim = len(bounds)
    dataset = {}
    dataset['test_x'] = np.zeros((dim, num))
    for i in range(dim):
        dataset['test_x'][i] = np.random.uniform(bounds[i][0], bounds[i][1], num)
    dataset['test_y'] = funct[1](dataset['test_x'])
    return dataset

def get_funct(funct):
    if funct == 'branin':
        return [branin_low, branin_high]
    else:
        return [branin_low, branin_high]

