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

def get_funct(funct):
    if funct == 'branin':
        return [branin_low, branin_high]
    elif funct == 'hartmann3d':
        return [hartmann3d_low, hartmann3d_high]
    elif funct == 'hartmann6d':
        return [hartmann6d_low, hartmann6d_high]
    else:
        return [branin_low, branin_high]
    


