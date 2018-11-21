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
 
def pump_charge_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    param_file = './test_bench/pump_charge/low_fidelity_circuit/param'
    result_file = './test_bench/pump_charge/low_fidelity_circuit/result.po'
    name = ['q_llower', 'q_wlower', 'q_lupper', 'q_wupper', 'q_lc', 'q_wc', 'q_lref', 'q_wref', 'q_lq', 'q_wq', 'lpdbin', 'wpdbin', 'lpdin', 'wpdin', 'luumid', 'wuumid', 'lumid', 'wumid', 'lp4', 'wp4', 'ln4', 'wn4', 'lnsupp', 'wnsupp', 'lnsupp2', 'wnsupp2', 'li10', 'wi10', 'lb1', 'wb1', 'lb2', 'wb2', 'lb3', 'wb3', 'lb4', 'wb4']
    ret = np.zeros((6, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(name)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        os.system('bash ./bash_file/pump_charge_low.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                ret[i, p] = float(line[i])
    return ret

def pump_charge_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    param_file = './test_bench/pump_charge/circuit/param'
    result_file = './test_bench/pump_charge/circuit/result.po'
    name = ['q_llower', 'q_wlower', 'q_lupper', 'q_wupper', 'q_lc', 'q_wc', 'q_lref', 'q_wref', 'q_lq', 'q_wq', 'lpdbin', 'wpdbin', 'lpdin', 'wpdin', 'luumid', 'wuumid', 'lumid', 'wumid', 'lp4', 'wp4', 'ln4', 'wn4', 'lnsupp', 'wnsupp', 'lnsupp2', 'wnsupp2', 'li10', 'wi10', 'lb1', 'wb1', 'lb2', 'wb2', 'lb3', 'wb3', 'lb4', 'wb4']
    ret = np.zeros((6, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(name)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        os.system('bash ./bash_file/pump_charge_high.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                ret[i, p] = float(line[i])
    return ret
    

def hartmann3d_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*0.0001
    alpha = np.array([1.0,1.2,3.0,3.2])
    ret = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:,i] - P)**2
        tmp = tmp.sum(axis=1)
        ret[0,i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def hartmann3d_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*0.0001
    alpha = np.array([1.0,1.2,3.0,3.2])
    theta = np.array([0.01,-0.01,-0.1,0.1])
    alpha = alpha + 2*theta
    ret = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:,i] - P)**2
        tmp = tmp.sum(axis=1)
        ret[0,i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def hartmann6d_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])*0.0001
    ret = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:, i]-P)**2
        tmp = tmp.sum(axis=1)
        ret[0, i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def hartmann6d_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    theta = np.array([0.01,-0.01,-0.1,0.1])
    alpha = alpha + 2*theta
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])*0.0001
    ret = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        tmp = A*(x[:, i]-P)**2
        tmp = tmp.sum(axis=1)
        ret[0, i] = -np.dot(alpha, np.exp(-tmp))
    return ret

def ellipsoid_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    dim = bounds.shape[0]
    y = ((x**2).T * np.arange(1,dim+1)).T.sum(axis=0)
    return y.reshape(1, -1)

def ellipsoid_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    dim = x.shape[0]
    s1 = np.array([0.3, 0.4, 0.2, 0.6, 1, 0.9, 0.2, 0.8, 0.5, 0.7, 0.4, 0.3, 0.7, 1, 0.9, 0.6, 0.2, 0.8, 0.2, 0.5])
    s2 = np.array([1.8, 0.4, 2, 1.2, 1.4, 0.6, 1.6, 0.2, 0.8, 1, 1.3, 1.1, 1.2, 1.4, 0.5, 0.3, 1.6, 0.7, 0.3, 1.9])
    y = ((x.T - s2)**2 * s1 * np.arange(1,dim+1)).T.sum(axis=0)
    return y.reshape(1, -1)

def Dixon_Price_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = (x[0]-1)**2
    for i in range(1,x.shape[0]):
        y = y + (i+1)*(2*x[i]**2 - x[i-1])**2
    return y.reshape(1, -1)

def Dixon_Price_low(x, bounds):
    delta = bounds[:,1] - bounds[:,0]
    s = np.array([1.8, 0.5, 2, 1.2, 0.4, 0.2, 1.4, 0.3, 1.6, 0.6, 0.8, 1, 1.3, 1.9, 0.7, 1.6, 0.3, 1.1, 1.2, 1.4])
    x = ((x.T * delta - s)/delta).T
    return Dixon_Price_high(x, bounds)

def Styblinski_Tang_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 0.5*(x**4 - 16*x**2 + 5*x).sum(axis=0)
    return y.reshape(1, -1)

def Styblinski_Tang_low(x, bounds):
    delta = bounds[:,1] - bounds[:,0]
    s = np.array([0.28, 0.59, 0.47, 0.16, 0.32])
    x = ((x.T * delta - s)/delta).T
    return Styblinski_Tang_high(x, bounds)

def Levy_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    dim = x.shape[0]
    w = 1 + 0.25*(x-1)
    y = np.sin(np.pi*w[0])**2
    y = y + ((w[:dim-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:dim-1]+1)**2)).sum(axis=0)
    y = y + (1 + np.sin(2*np.pi*w[dim-1])**2) * (w[dim-1]-1)**2
    return y.reshape(1, -1)

def Levy_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:, 1] - bounds[:, 0]
    x = (x.T * delta + mean).T
    dim = x.shape[0]
    sf = 0.8
    ss = np.array([1.2, 0.3, 1, 0.3, 1.6, 0.8, 1.4, 0.7, 2, 1.5])
    w = 1 + 0.25*(x.T - ss - 1).T
    y = np.sin(sf*np.pi*w[0])**2
    y = y + ((w[:dim-1] - 1)**2 * (1 + 10*np.sin(sf*np.pi*w[:dim-1]+1)**2)).sum(axis=0)
    y = y + (1 + np.sin(2*sf*np.pi*w[dim-1])**2) * (w[dim-1]-1)**2
    return y.reshape(1, -1)

def Ackley1_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt((x**2).mean(axis=0))) - np.exp(np.cos(2*np.pi*x).mean(axis=0))
    return y.reshape(1, -1)

def Ackley1_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    sf = 1.3
    ss = np.array([1.3, 0.1, 1.4, 0.8, 1.7, 1, 1.5, 0.6, 2, 0.4])
    y = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt(((x.T - ss)**2).mean(axis=1))) - np.exp(np.cos(2*sf*np.pi*x.T - ss).mean(axis=1))
    return y.reshape(1, -1)

def Ackley2_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    y = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt((x**2).mean(axis=0))) - np.exp(np.cos(2*np.pi*x).mean(axis=0))
    return y.reshape(1, -1)

def Ackley2_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    sf = 1.3
    ss = np.array([1.2, 0.2, 1.4, 0.8, 1.8, 1, 1.6, 0.6, 2, 0.4, 1.3, 0.3, 1.5, 0.9, 1.9, 1.1, 1.7, 0.7, 2.1, 0.5])
    y = 20 + np.exp(1) - 20 * np.exp(-0.2*np.sqrt(((x.T - ss)**2).mean(axis=1))) - np.exp(np.cos(2*sf*np.pi*x.T - ss).mean(axis=1))
    return y.reshape(1, -1)

def get_funct(funct):
    if funct == 'branin':
        return [branin_low, branin_high]
    elif funct == 'circuit1':
        return [circuit1_low, circuit1_high]
    elif funct == 'hartmann3d':
        return [hartmann3d_low, hartmann3d_high]
    elif funct == 'hartmann6d':
        return [hartmann6d_low, hartmann6d_high]
    elif funct == 'pump_charge':
        return [pump_charge_low, pump_charge_high]
    elif funct == 'Ackley1':
        return [Ackley1_low, Ackley1_high]
    elif funct == 'Ackley2':
        return [Ackley2_low, Ackley2_high]
    elif funct == 'ellipsoid':
        return [ellipsoid_low, ellipsoid_high]
    elif funct == 'Dixon_Price':
        return [Dixon_Price_low, Dixon_Price_high]
    elif funct == 'Styblinski_Tang':
        return [Styblinski_Tang_low, Styblinski_Tang_high]
    elif funct == 'Levy':
        return [Levy_low, Levy_high]
    else:
        return [branin_low, branin_high]
    


