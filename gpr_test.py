import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

# GPy.plotting.change_plotting_library('plotly')

# 1d model

np.random.seed(0)
X = np.random.uniform(-3.,3.,(50,1))
Y = np.sin(X) + np.random.randn(50,1)*0.05
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X_train,Y_train,kernel)
m.optimize(messages=False)

Y_test_pred, Y_test_var = m.predict(X_test)
print('test label: ', Y_test)
print('test pred: ', Y_test_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
print('mean squared error: ', mse_test)
r2_score_test = r2_score(Y_test, Y_test_pred)
print('r2 score: ', r2_score_test)

# fig = m.plot(plot_density=True)
# print(fig)
# fig.io.write_image('gpr_test.png')
# fig['gpmean'][0].savefig('gpr_test.png')
# GPy.plotting.show(fig, filename='basic_gp_regression_density_optimized_test')
# matplotlib.pylab.show(block=True) 
