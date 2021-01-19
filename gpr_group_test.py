import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from gpr_group_model import GPRegression_Group
from generate_data import generate_data_func
from plot import plot_1d, plot_2d

X_train_range_low = -3.
X_train_range_high = 3. 
X_test_range_low = -3.5
X_test_range_high = 3.5

# GPy.plotting.change_plotting_library('plotly')

# 1d model

# np.random.seed(1996)
# X = np.random.uniform(-3.,3.,(50,1))
# Y = np.sin(X) + np.random.randn(50,1)*0.05
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

num_train = 30
num_test = 100
# num_train = X_train.shape[0]
# TODO: for now, assume num_train/num_group is integer
num_group = 5
num_element_in_group = int(num_train/num_group)
dim = 2

# grouping choices: random, bins
grouping_method = 'random' 

# np.random.seed(1996)
# X_train = np.random.uniform(-3.,3.,(num_train,1))
# f_train = np.sin(X_train)
# Y_train = f_train + np.random.randn(num_train,1)*0.05
# X_test = np.random.uniform(-3.5,3.5,(num_test,1))
# f_test = np.sin(X_test)
# Y_test = f_test + np.random.randn(num_test,1)*0.05

X_train, f_train, Y_train, X_test, f_test, Y_test = generate_data_func(num_train,num_test,dim=dim)

# Generate group matrix A (10 * n_train) and group label Y_group (10 * 1)

idx_set = set(range(num_train))
A = np.zeros((num_group, num_train))

if grouping_method == 'random':
    # Method 1 -> form group: uniformly random 
    for i in range(num_group):
        select_ele = np.random.choice(list(idx_set), size = num_element_in_group, replace=False)
        A[i, np.asarray(select_ele)] = 1
        idx_set -=set(select_ele)
elif grouping_method == 'bins':
    # Method 2 -> form group: bins
    # TODO: do not work for 2d yet
    bins = np.linspace(X_train_range_low, X_train_range_high, num_group+1)
    digitized = np.digitize(X_train, bins)
    # print(digitized)
    for i in range(num_group):
        # print(i)
        # print(X_train[digitized == i])
        # print('size: ', len(X_train[digitized == i]))
        idx = np.asarray(digitized == i+1).reshape(X_train.shape[0],)
        A[i, idx] = 1
elif grouping_method == 'evenly':
    group_idx = 0
    sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
    for i in sorted_train_idx:
        A[group_idx, i] = 1
        if group_idx < num_group -1:
            group_idx +=1 
        else:
            group_idx = 0
    print(A.sum(axis=0))
    print(A.sum(axis=1))
else:
    print('invalid grouping method!')

Y_group = A.dot(Y_train)

kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)

m = GPRegression_Group(X_train,Y_group,kernel, A = A)
m.optimize(messages=False,max_f_eval = 1000)

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

# sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
# sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

# plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.',label = 'train')
# plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
# plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
# plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
#             (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
#             (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
# plt.legend()
# info = '_train_' + str(num_train) + '_group_' + str(num_group)
# plt.title('gprg' + info)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('gprg' + info + '.png')

if dim == 1:
    plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, 'gprg', num_group, grouping_method)  
if dim == 2:
    plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, 'gprg', num_group, grouping_method)
