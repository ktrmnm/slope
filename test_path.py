import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from slope import Slope, normal_decay

from sklearn import datasets

#np.random.seed(123)

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
#X = np.random.randn(20 * 50).reshape([20, 50]).astype(np.float64)
#theta = np.zeros(50, dtype=np.float64)
#theta[:5] = 2.0
#y = np.dot(X, theta)

#u = np.sort(np.abs(X.T @ y) / X.shape[0])[::-1]
#alpha_decay_ = normal_decay(X.shape[1])
#alpha_max = 2 * np.max(np.cumsum(u) / np.cumsum(alpha_decay_))
#print(alpha_max)

model_lasso = Lasso()
model_slope = Slope()

t1 = time.time()
alphas_lasso, coefs_lasso, gaps_lasso = model_lasso.path(X, y, l1_ratio=1.0, eps=1e-3)
t_lasso_path = time.time() - t1

t1 = time.time()
alphas_slope, coefs_slope, gaps_slope = model_slope.path(X, y, eps=1e-3, verbose=False)
t_slope_path = time.time() - t1

fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
axes[0].plot(alphas_lasso, coefs_lasso.T)
axes[0].set_title('Lasso path (time = {0:.2f})'.format(t_lasso_path))

axes[1].plot(alphas_slope, coefs_slope.T)
axes[1].set_title('Slope path (time = {0:.2f})'.format(t_slope_path))

#plt.xscale('log', basex=10)

plt.show()
