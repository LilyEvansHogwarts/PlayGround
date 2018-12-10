from activations import *
from NN import NN

dim = 2
layer_sizes = np.array([3,3])
activations = [get_act_f('relu')]*2
x = np.random.randn(dim, 3)

model = NN(layer_sizes, activations)
num_param = model.num_param(dim)
w = np.random.randn(num_param)
w_nobias = model.w_nobias(w, dim)
out = model.predict(w, x)
print(out)
