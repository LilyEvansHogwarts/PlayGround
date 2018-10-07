import sys
import toml
from get_dataset import *
import random
from src.Bagging import Bagging
from src.activations import *

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['name']
num_models = conf['num_models']
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

dataset['train_y'] = dataset['train_y']
dataset['test_y'] = dataset['test_y']

model = Bagging(name, num_models, dataset, bfgs_iter=max_iter[0], debug=True, layer_sizes=[layer_size[0]]*num_layers[0], activations=[get_act_f(activations[0])]*num_layers[0], l1=l1[0], l2=l2[0])
model.train(scale=scale)
py, ps2 = model.predict(dataset['test_x'])
print('py',py)
print('true',dataset['test_y'])
print('ps2',np.diag(ps2))
print('delta',dataset['test_y'] - py)
