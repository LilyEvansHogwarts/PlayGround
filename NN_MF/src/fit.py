from autograd import grad
import autograd.numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize
import traceback
import sys
from .NAR_BO import NAR_BO

def fit_low(x, model):
    x0 = np.copy(x).reshape(-1)
    best_loss = np.inf
    best_x = np.copy(x)

    def loss(x):
        nonlocal best_loss
        nonlocal best_x
        x = x.reshape(model.dim,-1)
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].predict_low(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[1,0])/ps
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            tmp1 = np.minimum(-6, tmp)**2
            EI2 = np.log(ps) - tmp1/2 - np.log(tmp1-1)
            EI = EI1*(tmp > -6) + EI2*(tmp <= -6)
        PI = np.zeros((x.shape[1]))
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].predict_low(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py/ps)
        tmp_loss = -EI-PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit low. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug)
        except:
            print('Fit low. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit low. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit low. Fail to build GP model')
        sys.exit(1)

    return best_x

def fit_high(x, model):
    x0 = np.copy(x).reshape(-1)
    best_loss = np.inf
    best_x = np.copy(x)

    def loss(x):
        nonlocal best_loss
        nonlocal best_x
        x = x.reshape(model.dim,-1)
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[1,0])/ps
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            tmp1 = np.minimum(-6, tmp)**2
            EI2 = np.log(ps) - tmp1/2 - np.log(tmp1-1)
            EI = EI1*(tmp > -6) + EI2*(tmp <= -6)
        PI = np.zeros((x.shape[1]))
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py/ps)
        tmp_loss = -EI-PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit high. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug)
        except:
            print('Fit high. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit high. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit high. Fail to build GP model')
        sys.exit(1)

    return best_x

def fit_py(x, model):
    x0 = np.copy(x).reshape(-1)
    best_loss = np.inf
    best_x = np.copy(x)

    def loss(x):
        nonlocal best_loss
        nonlocal best_x
        x = x.reshape(model.dim, -1)
        py, ps2 = model.models[0].predict(x)
        tmp_loss = py.sum()
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].predict(x)
            tmp_loss += np.maximum(py, 0).sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit py. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug)
        except:
            print('Fit py. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit py. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit py. Fail to build GP model')
        sys.exit(1)

    return best_x


