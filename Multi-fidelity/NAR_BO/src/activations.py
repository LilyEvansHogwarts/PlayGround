import autograd.numpy as np
import traceback
import sys
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b


def logphi(x):
    if x**2 < 0.0492:
        lp0 = -x/np.sqrt(2*np.pi)
        c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
        f = 0
        for i in range(14):
            f = lp0*(c[i]+f)
        return -2*f-np.log(2)
    elif x < -11.3137:
        r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
        q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
        num = 0.5641895835477550741
        for i in range(5):
            num = -x*num/np.sqrt(2)+r[i]
        den = 1.0
        for i in range(6):
            den = -x*den/np.sqrt(2)+q[i]
        return np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x**2)
    else:
        return np.log(0.5*np.maximum(0.000001,(1.0-erf(-x/np.sqrt(2)))))

'''
def logphi_vector(x):
    y = np.zeros((x.size))
    
    idx = (x**2 < 0.0492)
    lp0 = -x[idx]/np.sqrt(2*np.pi)
    c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
    f = 0
    for i in range(14):
        f = lp0*(c[i]+f)
    y[idx] = -2*f-np.log(2)
    
    idx = (x < -11.3137)
    r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
    q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
    num = 0.5641895835477550741
    for i in range(5):
        num = -x[idx]*num/np.sqrt(2)+r[i]
    den = 1.0
    for i in range(6):
        den = -x[idx]*den/np.sqrt(2)+q[i]
    y[idx] = np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x[idx]**2)
    
    idx = (x >= -11.3137)
    y[idx] = np.log(0.5*np.maximum(0.000001,(1.0-erf(-x[idx]/np.sqrt(2)))))
    return y
'''

# logphi_vector for autograd
def logphi_vector(x):
    # phi1
    lp0 = -x/np.sqrt(2*np.pi)
    c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
    f = 0
    for i in range(14):
        f = lp0*(c[i]+f)
    phi1 = -2*f - np.log(2)

    # phi2 
    r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
    q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
    num = 0.5641895835477550741
    for i in range(5):
        num = -x*num/np.sqrt(2)+r[i]
    den = 1.0
    for i in range(6):
        den = -x*den/np.sqrt(2)+q[i]
    phi2 = np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x**2)

    # phi3
    phi3 = np.log(0.5*np.maximum(0.000001,(1.0-erf(-x/np.sqrt(2)))))

    phi = phi1 * (x**2 < 0.0492) + phi2 * (x < -11.3137) + phi3 * ((x >= -11.3137) | (x**2 >= 0.0492))
    return phi

def get_act_f(act):
    act_f = relu
    if act == 'tanh':
        act_f = tanh
    elif act == 'sigmoid':
        act_f = sigmoid
    elif act == 'erf':
        act_f = erf
    return act_f

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pdf(x):
    # x = (x-mu)/theta
    return np.exp(-x**2 / 2)/np.sqrt(2*np.pi)


'''
from scipy.special import erf

def cdf(x, mu, theta):
    x = (x-mu)/theta
    return 0.5 + erf(np.sqrt(2)/2 * x)/2

# as erf from scipy.special won't work for autograd
# we decided to implement erf in autograd.numpy
'''
# the code reference: www.johndcook.com/blog/python_erf/

def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
                            
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
                                                        
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
                                                                    
    return sign*y

def cdf(x):
    # x = (x-mu)/theta
    return 0.5 + erf(x/np.sqrt(2))/2

'''
x0 = [0.5, 1.0, 2.0]

def loss(x):
    nlz = cdf(x[0], x[1], x[2])
    print x, nlz
    return nlz

gloss = grad(loss)

try:
    fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=100, iprint=1)
except np.linalg.LinAlgError:
    print('Increase noise term and re-optimization')
    x0 = 0.3
    try:
        fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=10, iprint=1)
    except:
        print('Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())
except:
    print('Exception caught, L-BFGS early stopping...')
    print(traceback.format_exc())
                
'''

