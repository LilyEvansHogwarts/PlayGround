# Gaussian Process

### cdf & pdf

* In order to furthur calculate the integration of **Probability Distribution Function(PDF)**, which is the **Cumulative Distribution Function(CDF)**, we import sympy for experiment.

```bash
>>> from sympy import *
>>> x = symbols('x')
>>> print(integrate(exp(-x**2 / 2)/sqrt(2*pi), (x, -1, 1)))
erf(sqrt(2)/2)
```
* Thus, we can get cdf(x) = 0.5 + erf(x/sqrt(2))/2.

```python
import numpy as np
from scipy.special import erf

def cdf(x):
    return 0.5 + erf(x/sqrt(2))/2

def pdf(x):
    return np.exp(-x**2 / 2)/np.sqrt(2 * np.pi) 
```
* Considering the fact that scipy.special doesn't work for autograd, we decide implement erf function in autograd.numpy. The code comes from website: <a href="www.johndcook.com/blog/python_erf/" target="_blank">erf function in numpy</a>

### debug

* remember to use `import autograd.numpy as np` instead of `import numpy as np` for furthur grad computation 
* there is no way to get `grad(np.diagonal)`. Considering the fact that ps2.shape=(self.num_test, self.num_test), we just reduce np.diagonal function. And we can get the same ps2 as before.
* autograd only works for function that has one parameter. Thus, we need to change `cdf(x, mu, theta)` and `pdf(x, mu, theta)` into `cdf(x)` and `pdf(x)`. Compute `x=(x-mu)/theta` before feed it to `cdf` or `pdf` function.
* autograd only works for function whose parameter.ndim is 1
* use `np.sqrt(np.sum())` instead of `np.sqrt().sum()` to greatly reduce the computation complexity

### results

#### pump charge

* python3 run.py toml/pump_charge.toml
* init = 100

| iteration | 100 | 200 | 300 | 400 | 500 | 600 | 689 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| node15 | 4.445 | 4.445 | 4.367 | 3.940 | 3.586 | 3.242 | 3.242 |
| node16 | 4.021 | 4.021 | 3.824 | 3.348 | 3.165 | 3.150 | 3.120 |
| node17 | 4.646 | 3.825 | 3.825 | 3.801 | 3.801 | 3.801 | 3.561 |
| node18 | 3.733 | 3.675 | 3.675 | 3.675 | 3.479 | 3.471 | 3.466 |
| node19 | 3.606 | 3.606 | 3.606 | 3.594 | 3.591 | 3.403 | 3.394 |
| node20 | 4.264 | 4.008 | 4.008 | 3.460 | 3.170 | 3.170 | 3.170 |
| node21 | 3.927 | 3.927 | 3.924 | 3.747 | 3.551 | 3.435 | 3.339 |
| node22 | 3.662 | 3.709 | 3.709 | 3.366 | 3.220 | 3.220 | 3.220 |
| node23 | 4.362 | 4.150 | 3.530 | 3.530 | 3.439 | 3.391 | 3.391 |
| node24 | 4.130 | 3.919 | 3.485 | 3.479 | 3.479 | 3.431 | 3.278 |
| node25 | 4.772 | 3.995 | 3.995 | 3.861 | 3.780 | 3.498 | 3.492 |
| node26 | 4.106 | 4.106 | 3.729 | 3.516 | 3.513 | 3.461 | 3.343 |

| average iteration | 62 | 105 | 193 | 316 | 462 | 534 | 618 |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| mean   | 4.140 | 3.949 | 3.806 | 3.610 | 3.481 | 3.389 | 3.335 |
| median | 4.096 | 3.968 | 3.966 | 3.604 | 3.361 | 3.303 | 3.255 |
| best   | 3.606 | 3.606 | 3.485 | 3.348 | 3.165 | 3.150 | 3.120 |
| worst  | 4.772 | 4.445 | 4.367 | 3.940 | 3.801 | 3.801 | 3.561 |

### tricks

* sample 20% points around the current best
* sample several dataset around the best of the last sample
* acquisition function: log(wEI)
* np.maximum(0.000001, ps2)
* initialize weights of the NN with MSE before constructing GP model
* preprocess the dataset, mean = 0, std = 1 for both input and output
