import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from gpr_group_model import GPRegression_Group
from generate_data import generate_data_func
from plot import plot_1d, plot_2d

x_shift = 20

X_train_range_low = -3. + x_shift
X_train_range_high = 3. + x_shift
X_test_range_low = -3.5 + x_shift
X_test_range_high = 3.5 + x_shift

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
num_group = 30
num_element_in_group = int(num_train/num_group)
dim = 1

# np.random.seed(1996)
# X_train = np.random.uniform(-3.,3.,(num_train,1))
# f_train = np.sin(X_train)
# Y_train = f_train + np.random.randn(num_train,1)*0.05
# X_test = np.random.uniform(-3.5,3.5,(num_test,1))
# f_test = np.sin(X_test)
# Y_test = f_test + np.random.randn(num_test,1)*0.05

X_train, f_train, Y_train, X_test, f_test, Y_test = generate_data_func(num_train,num_test,dim=dim)

# Generate group matrix A (10 * n_train) and group label Y_group (10 * 1)

def generate_A(grouping_method):

    # grouping choices: random, bins, evenly, ortho, tridiagonal, spd
    if grouping_method in {'ortho', 'tridiagonal', 'spd', 'diagonal'}:
        num_group = num_train
        print('Set the number of group as the same size of training samples.')

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
        print(bins)
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
    elif grouping_method == 'diagonal':
        A = np.eye(num_train)
    elif grouping_method == 'tridiagonal':
        from scipy.sparse import diags
        A = diags([1, 1], [-1, 1], shape=(num_group, num_train)).toarray()
        assert np.linalg.matrix_rank(A) == num_train
        assert np.linalg.matrix_rank(A) == num_train
        print(A)
    elif grouping_method == 'ortho':
        from scipy.stats import ortho_group
        A = ortho_group.rvs(num_train)
        assert np.linalg.matrix_rank(A) == num_train
        print(A)
    elif grouping_method == 'spd':
        from sklearn.datasets import make_spd_matrix
        A = make_spd_matrix(num_train)
        assert np.linalg.matrix_rank(A) == num_train
        print(A)
    else:
        print('invalid grouping method!')

    return A

# grouping_method = 'spd'

A_diagonal = generate_A('diagonal')
A_ortho = generate_A('ortho')
# print(A.sum(axis=0))
# print(A.sum(axis=1))

# for now, it is better to keep group aggregation over noiseless output and then add noise after group label is created. Since we do not want the noise level to be proportional to the group size, this introduce more things to deal with when we think of how to form a group.

def run_gprg(A):
    Y_group = A.dot(f_train) + np.random.randn(num_group,1)*0.05
    # Y_group = A.dot(Y_train)

    kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)

    m = GPRegression_Group(X_train,Y_group,kernel, noise_var=0.005, A = A)
    # m.optimize(messages=False,max_f_eval = 1000)
    # m.optimize_restarts(num_restarts = 10)

    Y_test_pred, Y_test_var = m.predict(X_test)

    return Y_test_pred, Y_test_var

Y_test_pred_dia, Y_test_var_dia = run_gprg(A_diagonal)
Y_test_pred_ortho, Y_test_var_ortho = run_gprg(A_ortho)

fig,ax = plt.subplots(1, 2, figsize = (10,5))
ax[0].hist(Y_test_pred_dia - Y_test_pred_ortho)
ax[0].set_title('Y_test_pred_diagonal - Y_test_pred_ortho hist')
ax[1].hist(Y_test_var_dia - Y_test_var_ortho)
ax[1].set_title('Y_test_var_diagonal - Y_test_var_ortho hist')
plt.show()

# print('test label: ', Y_test)
# print('test pred: ', Y_test_pred)
# print('test var: ', Y_test_var)
# mse_test = mean_squared_error(Y_test, Y_test_pred)
# print('mean squared error: ', mse_test)
# r2_score_test = r2_score(Y_test, Y_test_pred)
# print('r2 score: ', r2_score_test)

# if dim == 1:
#     plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A, 'gprg', num_group, grouping_method)  
# if dim == 2:
#     plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A, 'gprg', num_group, grouping_method)
