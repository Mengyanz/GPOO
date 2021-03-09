import numpy as np
from sklearn.model_selection import train_test_split

# 1d model

# np.random.seed(1996)
# X = np.random.uniform(-3.,3.,(50,1))
# Y = np.sin(X) + np.random.randn(50,1)*0.05
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

np.random.seed(1996)
# np.random.seed(0)

# x_shift = 0

# X_train_range_low = -3. + x_shift
# X_train_range_high = 3. + x_shift
# X_test_range_low = -3.5 + x_shift
# X_test_range_high = 3.5 + x_shift

# X_train_range_low = -10. + x_shift
# X_train_range_high = 10. + x_shift
# X_test_range_low = -10.5 + x_shift
# X_test_range_high = 10.5 + x_shift


def generate_data_func(num_train = 40, num_test = 100, dim = 1, func_type = 'sin', X_train_range_low = -3., X_train_range_high = 3., x_shift = 0):

    num_train = num_train
    num_test = num_test

    # uniformly discretize the continuous space
    X_train = np.random.uniform(X_train_range_low + x_shift, X_train_range_high+x_shift,(num_train,dim))
    X_test = np.random.uniform(X_train_range_low  +x_shift, X_train_range_high +x_shift,(num_test,dim))

    if dim == 1:
        if func_type == 'sin':
            f_train = np.sin(X_train)
        elif func_type == 'linear':
            f_train = 1.2 * X_train 
        Y_train = f_train + np.random.randn(num_train,1)*0.05
        if func_type == 'sin':
            f_test = np.sin(X_test)
        elif func_type == 'linear':
            f_test = 1.2 * X_test
        Y_test = f_test + np.random.randn(num_test,1)*0.05
    elif dim == 2:
        f_train = np.sin(X_train[:,0:1]) * np.sin(X_train[:,1:2])
        Y_train = f_train # + np.random.randn(num_train,1)*0.05
        f_test = np.sin(X_test[:,0:1]) * np.sin(X_test[:,1:2])
        Y_test = f_test # + np.random.randn(num_test,1)*0.05
    else:
        print('Invalid dim!')
        f_train = None
        Y_train = None
        f_test = None
        Y_test = None

    if num_test == num_train: 
        # if we select the testing set as the same as training set
        return X_train, f_train, Y_train
    else:
        return X_train, f_train, Y_train, X_test, f_test, Y_test

