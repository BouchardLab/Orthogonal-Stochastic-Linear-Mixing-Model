---
Orthogonal Stochastic Linear Mixing Model
---

This is the python implementation of the paper [Bayesian Inference in High-Dimensional Time-Series with the Orthogonal Stochastic Linear Mixing Model].
We propose a new regression framework to model multivariate output response data, which not only capture the complex input-dependent correlation across 
outputs, but also is effient for massive model and capable for single-trial analysis in neural data. Please refer our model for more details.

###System Requirement

We test our code with python 3.6 on Ubuntu 18.04 and our implementation relies on pytorch, hdf5storage, tqdm, scipy, copy and time. Please use pip or conda to install those dependencies.

For example
```{r}
pip install pytorch
```

###Run

The example to run OSLMM for dataset [Equity] can be
```{r}
python main.py
```
