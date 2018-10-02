import sys
import toml
from get_dataset import *
import random
from gaussian_process import GP

argv = sys.argv[1:]
conf = toml.load(argv[0])

scale = conf['scale']
max_iter = conf['max_iter']

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
theta = model.rand_theta(scale=scale)
print(model.neg_likelihood(theta))
model.train(theta)
py, ps2 = model.predict(test_x)
print('py',py.T)
print('true',test_y)
print('ps2',np.diag(ps2))
print('delta',test_y - py.T)
