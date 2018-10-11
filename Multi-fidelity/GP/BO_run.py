import autograd.numpy as np
import sys
import toml
from src.activations import *
from src.BO import BO
from src.fit import *
import multiprocessing
from get_dataset import *

argv = sys.argv[1:]
conf = toml.load(argv[0])

funct = get_funct(conf['funct'])
num = conf['num']
