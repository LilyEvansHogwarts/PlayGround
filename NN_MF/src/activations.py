import autograd.numpy as np

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# the code reference: www.johndcook.com/blog/python_erf/
def erf(x):
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = np.sign(x)
    x = np.abs(x)

    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return sign*y

def pdf(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

def cdf(x):
    return 0.5 + erf(x/np.sqrt(2))/2


def get_act_f(act):
    act_f = relu
    if act == 'tanh':
        act_f = tanh
    elif act == 'sigmoid':
        act_f = sigmoid
    elif act == 'erf':
        act_f = erf
    return act_f



