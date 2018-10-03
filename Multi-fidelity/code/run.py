import sys
import toml
from get_dataset import *
import random
from src.GP import GP
from src.NN_GP import NN_GP
from src.NN_scale_GP import NN_scale_GP
from src.activations import *
from src.Multifidelity_GP import Multifidelity_GP

argv = sys.argv[1:]
conf = toml.load(argv[0])

scale = conf['scale']
l1 = conf['l1']
l2 = conf['l2']
max_iter = conf['max_iter']
num_layers = conf['num_layers']
layer_size = conf['layer_size']
activations = conf['activation']

bounds = conf['bounds']
dim = conf['dim']
outdim = conf['outdim']
num_train = conf['num_train']
num_test = conf['num_test']
funct = conf['funct']
main_f = get_main_f(funct)

dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

for k in dataset.keys():
    print(k,dataset[k].shape)

train_x = dataset['train_x']
train_y = dataset['train_y']
test_x = dataset['test_x']
test_y = dataset['test_y']

model = GP(train_x, train_y, bfgs_iter=max_iter, debug=False)
# model = NN_GP(train_x, train_y, [layer_size]*num_layers, [get_act_f(activations)]*num_layers, l1=l1, l2=l2, bfgs_iter=max_iter, debug=True)
# model = NN_scale_GP(train_x, train_y, [layer_size]*num_layers, [get_act_f(activations)]*num_layers, l1=l1, l2=l2, bfgs_iter=max_iter, debug=True)
theta = model.rand_theta(scale=scale)
print(model.neg_likelihood(theta))
model.train(theta)
py, ps2 = model.predict(test_x)
print('py',py.T)
print('true',test_y)
print('ps2',np.diag(ps2))
print('delta',test_y - py.T)
