import numpy as np
import pickle
import os
import random

def ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    tmp1 = np.square(x).sum(axis=0)/x.shape[1]
    tmp2 = np.cos(c*x).sum(axis=0)/x.shape[1]
    out = a+np.exp(1)-a*np.exp(-b*np.sqrt(tmp1))-np.exp(tmp2)
    return out.reshape((1,x.shape[1]))

def pump_charge(x):
    conf_file = './test_bench/pump_charge/conf'
    param_file = './test_bench/pump_charge/circuit/param'
    result_file = './test_bench/pump_charge/circuit/result.po'
    name = []
    for l in open(conf_file, 'r'):
        l = l.strip().split(' ')
        if l[0] == 'des_var':
            name.append(l[1])

    y = np.zeros((6, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(name)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        os.system('bash pump_charge.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                y[i, p] = float(line[i])
    return y

def test2(x):
    conf_file = './test_bench/test2/conf'
    param_file = './test_bench/test2/circuit/param'
    result_file = './test_bench/test2/circuit/result.po'
    name = []
    for l in open(conf_file, 'r'):
        l = l.strip().split(' ')
        if l[0] == 'des_var':
            name.append(l[1])

    y = np.zeros((3, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(x)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        os.system('bash sim2.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                y[i, p] = float(line[i])
    '''
    z = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        z[0, i] = 1.2*y[0, i]+10*y[1, i]+1.6*y[2, i]
    '''
    return y

def circuit(x):
    # get param name
    conf_file = './test_bench/circuit/conf'
    param_file = './test_bench/circuit/circuit/param'
    result_file = './test_bench/circuit/circuit/result.po'
    name = []
    for l in open(conf_file, 'r'):
        l = l.strip().split(' ')
        if l[0] == 'des_var':
            name.append(l[1])
    
    y = np.zeros((7, x.shape[1])) 
    for p in range(x.shape[1]):
        # write out param file
        with open(param_file, 'w') as f:
            for i in range(len(x)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        # hspice simulation
        os.system('bash sim.sh')

        # get results
        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                y[i, p] = float(line[i])

    return y

def branin(x):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8*np.pi)
    y = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = 2*((x[1, i]-b*x[0, i]*x[0, i]+c*x[0, i]-r)**2) + s*(1-t)*np.cos(x[0, i]) + s
    return y

def optCase(x):
    y = np.zeros((2, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = x[0, i]*x[0, i]+(x[1, i]-1)*(x[1, i]-1)+(x[2, i]+1)*(x[2, i]+1)*(x[2, i]-1)*(x[2, i]+2)
        y[1, i] = x[2, i]-x[1, i]*x[1, i]
    return y

def game1(x):
    y = np.zeros((10, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = 5*x[:4, i].sum() - 5*np.square(x[:4, i]).sum() - x[:4, i].sum()
        y[1, i] = 2*x[0, i]+2*x[1, i]+x[9, i]+x[10, i]-10
        y[2, i] = 2*x[0, i]+2*x[2, i]+x[9, i]+x[11, i]-10
        y[3, i] = 2*x[1, i]+2*x[2, i]+x[10, i]+x[11, i]-10
        y[4, i] = -8*x[0, i]+x[9, i]
        y[5, i] = -8*x[1, i]+x[10, i]
        y[6, i] = -8*x[2, i]+x[11, i]
        y[7, i] = -2*x[3, i]-x[4, i]+x[9, i]
        y[8, i] = -2*x[5, i]-x[6, i]+x[10, i]
        y[9, i] = -2*x[7, i]-x[8, i]+x[11, i]
    return y

def get_eval(x,bounds):
    tmp_x = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp_x[i,j] = x[i,j]*(bounds[i][1]-bounds[i][0])/50 + bounds[i][0]
    return tmp_x


def get_dataset(main_f, num_train, num_test, dim, outdim, bounds):
    train_x = np.zeros((dim, num_train))
    train_y = np.zeros((outdim, num_train))
    x = np.zeros((dim, 1))
    i = 0
    while i < num_train:
        x = np.random.uniform(0,50,(dim,1))
        y = main_f(get_eval(x,bounds))
        if (y == np.inf).sum() > 0:
            continue
        train_x[:,i:i+1] = x
        train_y[:,i:i+1] = y
        i = i+1
    test_x = np.zeros((dim, num_test))
    test_y = np.zeros((outdim, num_test))
    i = 0
    while i < num_test:
        x = np.random.uniform(0,50,(dim,1))
        y = main_f(get_eval(x,bounds))
        if (y == np.inf).sum() > 0:
            continue
        test_x[:,i:i+1] = x
        test_y[:,i:i+1] = y
        i = i+1
    dataset = {}
    dataset['train_x'] = train_x
    dataset['train_y'] = train_y
    dataset['test_x'] = test_x
    dataset['test_y'] = test_y
    return dataset

def get_main_f(funct):
    if funct == 'game1':
        main_f = game1
    elif funct == 'optCase':
        main_f = optCase
    elif funct == 'branin':
        main_f = branin
    elif funct == 'circuit':
        main_f = circuit
    elif funct == 'test2':
        main_f = test2
    elif funct == 'pump_charge':
        main_f = pump_charge
    else:
        main_f = ackley
    return main_f


