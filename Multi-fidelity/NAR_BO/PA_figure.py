import numpy as np
import pickle
from pylab import *

with open('PA_2.pickle','rb') as f:
    dataset = pickle.load(f)

x = dataset['x'][2]

for i in range(3):
    plot(x, dataset['low_y'][i])
    plot(x, dataset['high_y'][i])
    show()

