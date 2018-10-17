# Multi-fidelity Gaussian Process

### codeing tricks

* remember to standardize each output idx in GP.py
* instead of standardize output in BO, do it in GP works much better, as there is no need to consider re_standardize in fitting, wEI , prediction, and best_x, best_y, best_constr
* add `neg_likelihood.sum()` can greatly improve the precision of model construction
* compute fit process with both model1 and model2 works may convergence much faster in some cases. However, if the minimum of low fidelity model divert from high fidelity greatly, the optimization process will suffer.
* there is no need to re_standardize output for both GP and NAR_BO. the prediction function is only used for fitting and wEI to query the next x0, not for predict the real value.

### NAR_GP.py predict function

* In `NAR_BO/src/NAR_GP.py` file, the **predict** function is change from:
```python
def predict(self, test_x):
    nsamples = 100
    num_test = test_x.shape[1]
    py1, ps21 = self.model1.predict(test_x)
    Z = np.random.multivariate_normal(py1, ps21, nsamples)
    if self.debug:
        print('Z.shape', Z.shape)
        print('Z[0,:].shape', Z[0,:].shape)
        print('Z[0,:][None,:].shape', Z[0,:][None,:].shape)

    x = np.tile(test_x, nsamples)
    x = np.concatenate((x, Z.reshape(1,-1)))
    py, ps2 = self.model2.predict(x)
    py = py.reshape(-1,num_test)
    ps2 = np.diag(ps2).reshape(-1,num_test).mean(axis=0) + py.var(axis=0)
    ps2 = np.abs(ps2)
    py = py.mean(axis=0)
    return py1, ps21, py, ps2
```
to 
```python
def predict(self, test_x):
    py1, ps2 = self.model1.predict(test_x)
    x = np.concatenate((test_x, py1.reshape(1,-1)))
    py, ps2 = self.model2.predict(x)
    return py1, ps21, py, ps2
```
* The reason is that original function is unable to compute gradient, thus, the **fit** function is written with evolution strategy or annealing simulation algorithm for local search. Both evolution and annealing simulation works slower than gradient descend. And the results **x0** is much worst than gradient descend.
*  In order to make further gradient descend possible, we reduced **Z** matrix to py1 just like the training process.
* The optimization process convergences much faster in this way. Also, the time consumption in each iteration is also reduced.
* It is worth notice that this prediction function is only being used in fitting. The original prediction method is still used in computing wEI.




