import autograd.numpy as np

def branin(x):
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    return (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def get_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    train_x = np.random.uniform(-0.5, 0.5, (dim, num))
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    tmp = (train_x.T * delta + mean).T
    train_y = funct(tmp)
    return train_x, train_y
    
