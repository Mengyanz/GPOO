import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

X_train_range_low = -3.
X_train_range_high = 3. 
X_test_range_low = -3.5
X_test_range_high = 3.5


def plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, model_name = 'gprg', num_group = None, grouping_method = 'random'):

    num_train = X_train.shape[0]

    sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
    sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

    plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.',label = 'train')
    plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
    plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
    plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
                (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
                (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
    plt.legend()
    if model_name == 'gprg':
        info = model_name + '_train_' + str(num_train) + '_group_' + str(num_group) + '_' + str(grouping_method)
    elif model_name == 'gpr':
        info = model_name + '_train_' + str(num_train)
    else:
        print('unknown model name!')
        info = 'None'
    
    plt.title(info)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(info + '_1d' + '.png')

def plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, model_name = 'gprg', num_group = None, grouping_method = 'random'):
    # counter plot, only for mean
    # refer https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py

    num_train = X_train.shape[0]

    # sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
    # sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

    # npts = 200
    ngridx = 100
    ngridy = 100
    # x = np.random.uniform(-3.5, 3.5, npts)
    # y = np.random.uniform(-3.5, 3.5, npts)
    # z = np.sin(x) * np.sin(y)
    x = X_test[:,0]
    y = X_test[:,1]
    z_test = np.sin(x) * np.sin(y)
    z_test_pred = Y_test_pred.reshape(len(z_test),)
    z_diff = np.abs(z_test - z_test_pred)
    print(z_test)
    print(z_diff)

    # Create grid values first.
    xi = np.linspace(X_test_range_low, X_test_range_high, ngridx)
    yi = np.linspace(X_test_range_low, X_test_range_high, ngridy)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z_diff)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')


    plt.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = plt.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
    plt.plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    plt.colorbar(cntr1)
    plt.xlabel('X[:,0]')
    plt.ylabel('X[:,1]')

    # plt.contour(X_train[:,0], X_train[:,1], Y_train,label = 'train')
    # plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
    # plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
    # plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
    #             (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
    #             (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
    # plt.legend()
    if model_name == 'gprg':
        info = model_name + '_train_' + str(num_train) + '_group_' + str(num_group) + '_' + str(grouping_method)
    elif model_name == 'gpr':
        info = model_name + '_train_' + str(num_train)
    else:
        print('unknown model name!')
        info = 'None'

    plt.title(info + ' (pred mean - true mean)')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.savefig(info + '_2d' + '.png')
    # plt.show()