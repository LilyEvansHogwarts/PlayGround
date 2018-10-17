import autograd.numpy as np
import os
import string

def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    total = num[0]+num[1]
    x = np.random.uniform(-0.5, 0.5, (dim, total))
    dataset = {}
    dataset['low_x'] = x[:,:num[0]]
    dataset['high_x'] = x[:,num[0]:]
    dataset['low_y'] = funct[0](x[:,:num[0]], bounds)
    dataset['high_y'] = funct[1](x[:,num[0]:], bounds)
    return dataset

def get_test(funct, num, bounds):
    dim = bounds.shape[1]
    dataset = {}
    dataset['test_x'] = np.random.uniform(-0.5, 0.5, (dim, num))
    dataset['test_y'] = funct[1](dataset['test_x'], bounds)
    return dataset

def branin_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    tmp1 = -1.275*np.square(x[0]/np.pi) + 5*x[0]/np.pi + x[1] - 6
    tmp2 = (10 - 5/(4*np.pi))*np.cos(x[0])
    ret = tmp1*tmp1 + tmp2 + 10
    return ret.reshape(1,-1)

def branin_low(x, bounds):
    tmp = (x.T - np.array([1.0/30, 11.0/30])).T
    tmp1 = branin_high(tmp, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = 10*np.sqrt(tmp1) + 2*(x[0]-0.5) - 3*(x[1]-1) - 1
    return ret.reshape(1,-1)

def circuit1_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    param_file = './test_bench/circuit1_low/origin_circuit/param'
    result_file = './test_bench/circuit1_low/origin_circuit/result.po'
    name = ['cp', 'cs', 'w', 'vdd', 'vb']
    ret = np.zeros((3, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(name)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')
        
        os.system('bash ./bash_file/circuit1_low.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            line[0] = -float(line[0])
            line[1] = float(line[1]) - 13.68
            line[2] = 23.00 - float(line[2])
            ret[:, p] = line
    return ret
                    
def circuit1_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    param_file = './test_bench/circuit1_high/origin_circuit/param'
    result_file = './test_bench/circuit1_high/origin_circuit/result.po'
    name = ['cp', 'cs', 'w', 'vdd', 'vb']
    ret = np.zeros((3, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(name)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')
        
        os.system('bash ./bash_file/circuit1_high.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            line[0] = -float(line[0])
            line[1] = float(line[1]) - 13.68
            line[2] = 23.00 - float(line[2])
            ret[:, p] = line
    return ret
                    
def get_funct(funct):
    if funct == 'branin':
        return [branin_low, branin_high]
    elif funct == 'circuit1':
        return [circuit1_low, circuit1_high]
    else:
        return [branin_low, branin_high]
    


