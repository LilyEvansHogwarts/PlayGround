import hues
import autograd.numpy as np

def print_out(test_y, py, ps2):
    hues.info('test_y')
    print(test_y)
    hues.info('prediction')
    print(py)
    hues.info('uncertainty')
    if ps2.ndim == 1:
        print(np.sqrt(ps2))
    else:
        print(np.sqrt(np.diag(ps2)))
    delta = test_y - py
    # hues.info('delta')
    # print(delta)
    SE = np.dot(delta, delta.T)
    hues.success('SE', np.dot(delta, delta.T))
    SMSE = np.sqrt(np.dot(delta, delta.T)/delta.size)
    hues.success('SMSE', SMSE)
    


