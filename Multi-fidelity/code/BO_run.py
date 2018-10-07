import autograd.numpy as np
import sys
import toml
from src.BO import BO
from get_dataset import *
from src.activations import *

argv = sys.argv[1:]
conf = toml.load(argv[0])

num_train = conf['num_train']
num_test = conf['num_test']
funct = conf['funct']
dim = conf['dim']
outdim = conf['outdim']
bounds = conf['bounds']

main_f = get_main_f(funct)
dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

name = conf['name']
num_models = conf['num_models']
bfgs_iter = conf['max_iter']
num_layers = conf['num_layers']
layer_sizes = conf['layer_size']
activations = conf['activation']
l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']

for i in dataset.keys():
    print(i, dataset[i].shape)
tmp = np.copy(dataset)


model = BO(name, num_models, dataset, bfgs_iter=bfgs_iter, debug=False, scale=scale, num_layers=num_layers, layer_sizes=layer_sizes, activations=activations, l1=l1, l2=l2)
pys, ps2s = model.predict(dataset['test_x'])
print('pys', pys.T)
print('ps2s', ps2s.T)
print('true', dataset['test_y'])
delta = pys.T - dataset['test_y']
print('delta', delta)
print('MSE', np.dot(delta, delta.T))

