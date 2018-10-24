import autograd.numpy as np
from autograd import grad
from .BO import BO
import sys
import traceback
from scipy.optimize import fmin_l_bfgs_b, minimize
from .activations import *

def fit(x, model):
    x0 = np.copy(x)
    best_x = np.copy(x)
    best_loss = np.inf

    '''
    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.ones((x.shape[1]))
        if model.best_constr <= 0:
            py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[0])/ps
            EI = ps*(tmp*cdf(tmp)+pdf(tmp))
        PI = np.ones((x.shape[1]))
        for i in range(1,model.outdim):
            py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI*cdf(-py/ps)
        tmp_loss = -EI*PI
        tmp_loss = tmp_loss.sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss.sum()
            best_x = np.copy(x)
        return tmp_loss
    '''

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        x = x.reshape(model.dim, int(x.size/model.dim))
        EI = np.zeros((x.shape[1]))
        if model.best_constr <= 0:
            py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[0])/ps
            # tmp > -40
            # tmp1 = np.maximum(-40, tmp)
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            EI1 = np.log(np.maximum(0.000001, EI1))
            # tmp <= -40
            tmp2 = np.minimum(-40, tmp)**2
            EI2 = np.log(ps) - tmp2/2 - np.log(tmp2-1)
            # EI
            EI = EI1*(tmp > -40) + EI2*(tmp <= -40)
        PI = np.zeros((x.shape[1]))
        for i in range(1, model.outdim):
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
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=500, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit. Increase noise term and re-optimization')
        x0 = np.copy(best_x)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*model.dim, maxiter=500, m=10, iprint=model.debug)
        except:
            print('Fit. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit. Fail to buildGP model')
        sys.exit(1)

    return best_x


def fit_py(x, model, name):
    x0 = np.copy(x).reshape(-1)

    def get_py(idx):
        def loss(x0):
            x0 = x0.reshape(model.dim, int(x.size/model.dim))
            py, ps2 = model.models[idx].predict(x0)
            if idx == 0:
                py = py.sum()
            else:
                py = -py.sum()
            return py
        return loss
    
    
    if name == 'circuit1':
        constr = ({'type':'ineq', 'fun':get_py(1), 'jac':grad(get_py(1))},\
                {'type':'ineq', 'fun':get_py(2), 'jac':grad(get_py(2))})
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), constraints=constr, bounds=[[-0.5, 0.5]]*model.dim, method='SLSQP')
    elif name == 'branin' or name == 'hartmann3d' or name == 'hartmann6d':
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), bounds=[[-0.5, 0.5]]*model.dim, method='SLSQP')
    elif name == 'pump_charge':
        constr = ({'type':'ineq', 'fun':get_py(1), 'jac':grad(get_py(1))},\
                {'type':'ineq', 'fun':get_py(2), 'jac':grad(get_py(2))},\
                {'type':'ineq', 'fun':get_py(3), 'jac':grad(get_py(3))},\
                {'type':'ineq', 'fun':get_py(4), 'jac':grad(get_py(4))},\
                {'type':'ineq', 'fun':get_py(5), 'jac':grad(get_py(5))})
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), constraints=constr, bounds=[[-0.5, 0.5]]*model.dim, method='SLSQP')

    
    if np.isnan(data.x[0]):
        return np.zeros(x0.shape)
    else:
        return data.x
    
    
def fit_new_py(x, model):
    x0 = np.copy(x).reshape(-1)
    best_x = np.copy(x)
    best_loss = np.inf

    def loss(x0):
        nonlocal best_x
        nonlocal best_loss
        x0 = x0.reshape(model.dim, -1)
        py, ps2 = model.models[0].predict(x0)
        tmp_loss = py.sum()
        for i in range(1, model.outdim):
            py, ps2 = model.models[0].predict(x0)
            tmp_loss += np.maximum(0, py).sum()
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x0)
        return tmp_loss
    
    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug)
    except np.linalg.LinAlgError:
        print('Fit_new_py. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug)
        except:
            print('Fit_new_py. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit_new_py. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    if(np.isnan(best_loss) or np.isinf(best_loss)):
        print('Fit_new_py. Fail to build GP model')
        sys.exit(1)

    return best_x



