import autograd.numpy as np
import sys
import traceback
from autograd import grad
# from scipy.optimize import anneal
from .activations import *
from .NAR_BO import NAR_BO
import cma

# tmp_loss is scalar
def fit(x, model):
    x0 = np.copy(x).reshape(-1)
    best_x = np.copy(x)
    best_loss = np.inf

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = 1.0
        if model.best_constr[1] <= 0:
            _, _, py, ps2 = model.models[0].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            tmp = (model.best_y[1,0] - py)/ps
            if tmp > -40:
                EI = ps*(tmp*cdf(tmp)+pdf(tmp))
                EI = np.log(np.maximum(0.000001, EI))
            else:
                tmp2 = tmp**2
                EI = np.log(ps)-tmp2/2-np.log(tmp2-1)
        PI = 1.0
        for i in range(1,model.outdim):
            _, _, py, ps2 = model.models[i].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            PI = PI + logphi(-py/ps)
        tmp_loss = -EI-PI
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    xopt, es = cma.fmin2(loss, x0, 0.25, options={'maxfevals':50})
    # xopt = anneal(loss, x0, maxiter=50, lower=-10, upper=10, disp=True)
    return xopt.reshape(model.dim, int(xopt.size/model.dim))
    
def fit_test(x, model):
    x0 = np.copy(x).reshape(-1)

    def loss(x):
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            _, _, py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(ps2)
            tmp = -(py - model.best_y[1,0])/ps
            idx = (tmp > -40)
            EI[idx] = ps[idx]*(tmp[idx]*cdf(tmp[idx])+pdf(tmp[idx]))
            idx = (tmp <= -40)
            tmp2 = tmp[idx]**2
            EI[idx] = np.log(ps[idx]) - tmp2/2 - np.log(tmp2-1)
        PI = np.zeros((x.shape[1]))
        for i in range(1,model.outdim):
            _, _, py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(ps2)
            PI = logphi(-py/ps) + PI
        loss = -EI-PI
        return loss.min()

    xopt, es = cma.fmin2(loss, x0, 0.1, options={'maxiter':1, 'bounds':[-0.5,0.5]})
    # xopt = anneal(loss, x0, maxiter=50, lower=-10, upper=10, disp=True)
    return xopt.reshape(model.dim, int(xopt.size/model.dim))
    

