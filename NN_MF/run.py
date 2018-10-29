import autograd.numpy as np
from src.NAR_BO import NAR_BO
import sys
import toml


model = NAR_BO(num_models, dataset, gamma, scale, bounds, bfgs_iter, debug=False)
