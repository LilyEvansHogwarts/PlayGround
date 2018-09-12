import autograd.numpy as np
import sys
import traceback
from autograd import grad
from .Bagging_Constr_model import Bagging_Constr_model
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
        EI = 1.0
        if model.best_constr <= 0:
            py, ps2 = model.model[0].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            tmp = (model.best_y - py)/ps
            if tmp > -40:
                EI = ps*(tmp*cdf(tmp)+pdf(tmp))
                EI = np.log(np.maximum(0.000001, EI))
            else:
                EI = np.log(ps)-tmp*tmp/2-np.log(tmp*tmp-1)
        PI = 1.0
        for i in range(1, model.outdim):
            py, ps2 = model.model[i].predict(x)
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            # PI = PI*cdf(-py/ps)
            PI = PI + logphi(-py/ps)
        tmp_loss = -EI-PI
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)
    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=model.bounds, maxiter=200, m=100, iprint=0)
    except np.linalg.LinAlgError:
        print('Increase noise term and re-optimization')
        x0 = np.copy(best_x)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=model.bounds, maxiter=200, m=10, iprint=0)
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fail to build GP model')
        sys.exit(1)

    return best_x



