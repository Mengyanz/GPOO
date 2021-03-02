import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from generate_data import generate_data_func
from plot import plot_1d, plot_2d

# GPy.plotting.change_plotting_library('plotly')

# 1d model

# np.random.seed(0)
# X = np.random.uniform(-3.,3.,(50,1))
# Y = np.sin(X) + np.random.randn(50,1)*0.05
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

num_train = 15
num_test = 100
dim = 1

X_train, f_train, Y_train, X_test, f_test, Y_test = generate_data_func(num_train,num_test,dim=dim, func_type='linear')

kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
# kernel = GPy.kern.Matern52(dim,ARD=True) + GPy.kern.White(dim)

m = GPy.models.GPRegression(X_train,Y_train,kernel, noise_var=0.005)
# m.optimize(messages=False)

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

# plt.plot(X_train, Y_train, '.', label = 'train')
# plt.plot(Y_test, Y_test, '.', label = 'test')
# plt.plot(Y_test, Y_test_pred, '.', label = 'pred')
# plt.legend()
# plt.show()

# sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
# sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

# plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.',label = 'train')
# plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
# plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
# plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
#             (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
#             (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
# plt.legend()
# info = '_train_' + str(num_train) 
# plt.title('gpr' + info)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('gpr' + info + '.png')

if dim == 1:
    plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, None, 'gpr')
if dim == 2:
    plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, 'gpr')
