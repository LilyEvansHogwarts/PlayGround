import autograd.numpy as np
from autograd import grad, value_and_grad
import traceback
import sys
from .activations import *
from .NAR_BO import NAR_BO
import cma
from scipy.optimize import minimize, fmin_l_bfgs_b

# tmp_loss is scalar
def fit(x, model):
    best_x = np.copy(x)
    x0 = np.copy(x).reshape(-1)
    best_loss = np.inf
    tmp_loss = np.inf
    
    # loss log
    def loss(x):
        nonlocal tmp_loss
        x = x.reshape(model.dim, -1)
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].predict_low(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[0, 0])/ps
            # tmp > -6
            # tmp1 = np.maximum(-6, tmp)
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            EI1 = np.log(np.maximum(0.000001, EI1))
            # tmp <= -6
            tmp2 = np.minimum(-6, tmp)**2
            EI2 = np.log(ps) - tmp2/2 - np.log(tmp2-1)
            # EI
            EI = EI1*(tmp > -6) + EI2*(tmp <= -6)
        PI = np.zeros((x.shape[1]))
        for i in range(1, model.outdim):
            py, ps2 = model.models[i].predict_low(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py/ps)
            '''
            tmp = -py/ps
            # tmp > -6
            PI1 = np.log(cdf(tmp))
            # tmp <= -6
            tmp2 = np.minimum(-6, tmp)
            PI2 = -0.5*tmp2**2 - np.log(-tmp2) - 0.5*np.log(2*np.pi)
            PI = PI + PI1*(tmp > -6) + PI2*(tmp <= -6)
            '''
        tmp_loss = -EI-PI
        tmp_loss = tmp_loss.sum()
        return tmp_loss

    def callback(x):
        nonlocal best_x
        nonlocal best_loss
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
    
    # gloss = grad(loss)
    gloss = value_and_grad(loss)
    
    try:
        fmin_l_bfgs_b(gloss, x0, bounds=[[-0.5,0.5]]*x0.size, maxiter=2000, m=100, iprint=model.debug, callback=callback)
    except np.linalg.LinAlgError:
        print('fit, Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5,0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug, callback=callback)
        except:
            print('fit, Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('fit, Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())
    

    return best_x

    
def fit_test(x, model):
    best_x = np.copy(x)
    x0 = np.copy(x).reshape(-1)
    best_loss = np.inf
    tmp_loss = np.inf
    
    
    # loss log
    def loss(x):
        nonlocal tmp_loss
        x = x.reshape(model.dim, -1)
        EI = np.zeros((x.shape[1]))
        if model.best_constr[1] <= 0:
            py, ps2 = model.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            tmp = -(py - model.best_y[1, 0])/ps
            # tmp > -6
            # tmp1 = np.maximum(-6, tmp)
            EI1 = ps*(tmp*cdf(tmp)+pdf(tmp))
            EI1 = np.log(np.maximum(0.000001, EI))
            # tmp <= -6
            tmp2 = np.minimum(-6, tmp)**2
            EI2 = np.log(ps) - tmp2/2 - np.log(tmp2-1)
            # EI
            EI = EI1 * (tmp > -6) + EI2 * (tmp <= -6)
        PI = np.zeros((x.shape[1]))
        for i in range(1, model.outdim):
            py, ps2 = model.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py/ps)
            '''
            tmp = -py/ps
            # tmp > -6
            PI1 = np.log(cdf(tmp))
            tmp2 = np.minimum(-6, tmp)
            PI2 = -0.5*tmp2**2 - np.log(-tmp2) - 0.5*np.log(2*np.pi)
            PI = PI + PI1*(tmp > -6) + PI2*(tmp <= -6)
            '''
        tmp_loss = -EI-PI
        tmp_loss = tmp_loss.sum()
        return tmp_loss

    def callback(x):
        nonlocal best_x
        nonlocal best_loss
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)

    gloss = value_and_grad(loss)

    try:
        fmin_l_bfgs_b(gloss, x0, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug, callback=callback)
    except np.linalg.LinAlgError:
        print('Fit test. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug, callback=callback)
        except:
            print('Fit test. Exception caught,  L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit test. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())


    return best_x

def fit_py(x, model, name):
    x0 = np.copy(x).reshape(-1)

    def get_py(idx):
        def loss(x0):
            x0 = x0.reshape(model.dim, -1)
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
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), constraints=constr, bounds=[[-0.5, 0.5]]*x.size, method='SLSQP')
    elif name == 'branin' or name == 'hartmann3d' or name == 'hartmann6d':
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), bounds=[[-0.5, 0.5]]*x.size, method='SLSQP')
    elif name == 'pump_charge':
        constr = ({'type':'ineq', 'fun':get_py(1), 'jac':grad(get_py(1))},\
                {'type':'ineq', 'fun':get_py(2), 'jac':grad(get_py(2))},\
                {'type':'ineq', 'fun':get_py(3), 'jac':grad(get_py(3))},\
                {'type':'ineq', 'fun':get_py(4), 'jac':grad(get_py(4))},\
                {'type':'ineq', 'fun':get_py(5), 'jac':grad(get_py(5))})
        data = minimize(get_py(0), x0, jac=grad(get_py(0)), constraints=constr, bounds=[[-0.5, 0.5]]*x.size, method='SLSQP')
        
    
    if np.isnan(data.x[0]):
        return np.zeros(x.shape)
    else:
        return data.x.reshape(model.dim, -1)
    
def fit_new_py(x, model):
    x0 = np.copy(x).reshape(-1)
    best_x = np.copy(x)
    best_loss = np.inf
    tmp_loss = np.inf

    def loss(x0):
        nonlocal tmp_loss
        x0 = x0.reshape(model.dim, -1)
        py, ps2 = model.models[0].predict(x0)
        tmp_loss = py.sum()
        for i in range(1, model.outdim):
            py, ps2 = model.models[0].predict(x0)
            tmp_loss += np.maximum(0, py).sum()
        return tmp_loss

    def callback(x):
        nonlocal best_loss
        nonlocal best_x
        if tmp_loss < best_loss:
            best_loss = tmp_loss
            best_x = np.copy(x)
    
    gloss = value_and_grad(loss)

    try:
        fmin_l_bfgs_b(gloss, x0, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=100, iprint=model.debug, callback=callback)
    except np.linalg.LinAlgError:
        print('Fit_new_py. Increase noise term and re-optimization')
        x0 = np.copy(best_x).reshape(-1)
        x0[0] += 0.01
        try:
            fmin_l_bfgs_b(gloss, x0, bounds=[[-0.5, 0.5]]*x.size, maxiter=2000, m=10, iprint=model.debug, callback=callback)
        except:
            print('Fit_new_py. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Fit_new_py. Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    return best_x

