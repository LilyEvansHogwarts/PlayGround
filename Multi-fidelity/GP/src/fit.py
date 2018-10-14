import autograd.numpy as np
from autograd import grad
from .BO import BO
import sys
import traceback
from scipy.optimize import fmin_l_bfgs_b
from .activations import *

def fit(x, model):
    x0 = np.copy(x)
    best_x = np.copy(x)
    best_loss = np.inf

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.zeros((x.shape[1]))
        if model.best_constr <= 0:
            py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(np.diag(ps2)) + 0.000001
            # ps = np.maximum(0.000001, ps)
            tmp = -(py - model.best_y[0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
            '''
            idx = (tmp>-40)
            EI[idx] = ps[idx]*(tmp[idx]*cdf(tmp[idx])+pdf(tmp[idx]))
            EI[idx] = np.log(np.maximum(0.000001, EI[idx]))
            idx1 = (tmp<=-40)
            tmp[idx1] = tmp[idx1]**2
            EI[idx1] = np.log(ps[idx1]) - tmp[idx1]/2 - np.log(tmp[idx1]-1)
            '''
        PI = np.zeros((x.shape[1]))
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(np.diag(ps2))
            # PI = PI + logphi(-py/ps)
            PI = PI*cdf(-py/ps)
        # tmp_loss = -EI-PI
        tmp_loss = -EI*PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss.sum()
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=100, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Increase noise term and re-optimization')
        x0 = np.copy(best_x)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*model.dim, maxiter=100, m=10, iprint=model.debug)
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fail to buildGP model')
        sys.exit(1)

    return best_x