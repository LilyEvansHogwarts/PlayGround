import autograd.numpy as np
import sys
import toml
from src.NAR_BO import NAR_BO
from src.activations import *
from src.fit import fit
import multiprocessing
from get_dataset import *

argv = sys.argv[1:]
conf = toml.load(argv[0])

funct = get_funct(conf['funct'])
num = conf['num']
bounds = conf['bounds']
num_models = conf['num_models']
scale = conf['scale']
bfgs_iter = conf['bfgs_iter']

dataset = init_dataset(funct, num, bounds)
for i in dataset.keys():
    print(i, dataset[i].shape)

model = NAR_BO(num_models, dataset, scale, bounds, bfgs_iter=bfgs_iter, debug=True)

test = get_test(funct, 200, bounds)
test_x = test['test_x']
test_y = test['test_y']
_, _, py, ps2 = model.predict(test_x)

print('py', py)
print('ps2', ps2)
print('true', test_y)
delta = test_y - py
print('delta', delta)
print('MSE', np.dot(delta, delta.T))
