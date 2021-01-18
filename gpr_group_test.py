import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from gpr_group_model import GPRegression_Group

# GPy.plotting.change_plotting_library('plotly')

# 1d model

# np.random.seed(1996)
# X = np.random.uniform(-3.,3.,(50,1))
# Y = np.sin(X) + np.random.randn(50,1)*0.05
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

num_train = 40
num_test = 100
# num_train = X_train.shape[0]
num_group = 8
num_element_in_group = int(num_train/num_group)

np.random.seed(1996)
X_train = np.random.uniform(-3.,3.,(num_train,1))
f_train = np.sin(X_train)
Y_train = f_train + np.random.randn(num_train,1)*0.05
X_test = np.random.uniform(-3.5,3.5,(num_test,1))
f_test = np.sin(X_test)
Y_test = f_test + np.random.randn(num_test,1)*0.05


# Generate group matrix A (10 * n_train) and group label Y_group (10 * 1)

idx_set = set(range(num_train))
A = np.zeros((num_group, num_train))


for i in range(num_group):
    select_ele = np.random.choice(list(idx_set), size = num_element_in_group, replace=False)
    A[i, np.asarray(select_ele)] = 1
    idx_set -=set(select_ele)
Y_group = A.dot(Y_train)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPRegression_Group(X_train,Y_group,kernel, A = A)
m.optimize(messages=False)

Y_test_pred, Y_test_var = m.predict(X_test)
print('test label: ', Y_test)
print('test pred: ', Y_test_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
print('mean squared error: ', mse_test)
r2_score_test = r2_score(Y_test, Y_test_pred)
print('r2 score: ', r2_score_test)

# m.plot(plot_density=True)

# print(fig)
# fig.io.write_image('gpr_test.png')
# fig['gpmean'][0].savefig('gpr_test.png')
# GPy.plotting.show(fig, filename='basic_gp_regression_density_optimized_test')
# matplotlib.pylab.show(block=True) 

sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.',label = 'train')
plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
            (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
            (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
plt.legend()
info = '_train_' + str(num_train) + '_group_' + str(num_group)
plt.title('gprg' + info)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('gprg' + info + '.png')

