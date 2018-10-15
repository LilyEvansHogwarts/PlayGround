import autograd.numpy as np
from autograd import grad
import traceback
import sys
from .activations import *
from .NAR_BO import NAR_BO
import cma

# tmp_loss is scalar
def fit(x, model):
    best_x = np.copy(x)
    best_loss = np.inf
    x0 = np.copy(x).reshape(-1)

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.ones((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].predict_low(x)
            ps = np.sqrt(np.abs(np.diag(ps2))) + 0.000001
            # ps = np.maximum(0.000001, ps)
            tmp = -(py - model.best_y[0, 0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.zeros((x.shape[1]))
        for i in range(1, model.outdim):
            py,  ps2 = model.models[i].predict_low(x)
            ps = np.sqrt(np.abs(np.diag(ps2))) + 0.000001
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
        print('fit, Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*model.dim, maxiter=100, m=10, iprint=model.debug)
        except:
            print('fit, Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('fit, Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('fit, Fail to buildGP model')
        sys.exit(1)

    return best_x

    
def fit_test(x, model):
    best_x = np.copy(x)
    best_loss = np.inf
    x0 = np.copy(x).reshape(-1)

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.ones((x.shape[1]))
        if model.best_constr[1] <= 0:
            _,  _, py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(np.abs(np.diag(ps2))) + 0.000001
            tmp = -(py - model.best_y[1, 0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.ones((x.shape[1]))
        for i in range(1, model.outdim):
            _,  _, py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(np.abs(np.diag(ps2))) + 0.000001
            PI = PI*cdf(-py/ps)
        tmp_loss = -EI*PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
        return tmp_loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5, 0.5]]*x.size, maxiter=500, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit test. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5, 0.5]]*x.size, maxiter=500, m=10, iprint=model.debug)
        except:
            print('Fit test. Exception caught,  L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit test. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit test. Fail to build GP model')
        sys.exit(1)

    return best_x


