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
        EI = np.ones((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].model1.predict(x)
            ps = np.sqrt(np.diag(ps2))
            ps = np.maximum(0.000001,ps)
            tmp = -(py - model.best_y[0,0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.ones((x.shape[1]))
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].model1.predict(x)
            ps = np.sqrt(ps2.sum())
            PI = PI*cdf(-py/ps)
        tmp_loss = -EI*PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
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


def fit_test(x, model):
    x0 = np.copy(x).reshape(-1)

    def loss(x):
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            _, _, py, ps2 = model.models[0].predict(x)
            ps = np.maximum(0.000001,np.sqrt(ps2))
            tmp = -(py - model.best_y[1,0])/ps
            idx = (tmp > -40)
            EI[idx] = ps[idx]*(tmp[idx]*cdf(tmp[idx])+pdf(tmp[idx]))
            idx = (tmp <= -40)
            tmp2 = tmp[idx]**2
            EI[idx] = np.log(ps[idx]) - tmp2/2 - np.log(tmp2-1)
        PI = np.zeros((x.shape[1]))
        for i in range(1,model.outdim):
            _, _, py, ps2 = model.models[i].predict(x)
            ps = np.maximum(0.000001,np.sqrt(ps2))
            PI = logphi(-py/ps) + PI
        loss = -EI-PI
        return loss.min()

    xopt, es = cma.fmin2(loss, x0, 0.1, options={'maxiter':1, 'bounds':[-0.5,0.5], 'verb_disp':0})
    # xopt = anneal(loss, x0, maxiter=50, lower=-10, upper=10, disp=True)
    return xopt.reshape(model.dim, int(xopt.size/model.dim))
    

