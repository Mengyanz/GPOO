import matplotlib
import pretty_errors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.cm import ScalarMappable as sm
import numpy as np

x_shift = 20

X_train_range_low = -3. + x_shift
X_train_range_high = 3. + x_shift
X_test_range_low = -3.5 + x_shift
X_test_range_high = 3.5 + x_shift


def plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A, model_name = 'gprg', num_group = None, grouping_method = 'random'):

    num_train = X_train.shape[0]

    sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
    sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

    print('test label: ', Y_test[sorted_test_idx])
    print('test pred: ', Y_test_pred[sorted_test_idx])
    print('test std: ', np.sqrt(Y_test_var[sorted_test_idx]))

    if A is None:
        plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.', color = 'tab:green', label = 'train')
    else:
        # plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.', color = 'tab:green', label = 'train')
        A_sorted = A[:,sorted_train_idx]
        for i in range(A.shape[0]):
            plt.plot(X_train[sorted_train_idx,:][A_sorted[i,:] == 1,:], Y_train[sorted_train_idx,:][A_sorted[i,:] == 1,:], '.')
    plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], color = 'tab:orange', label = 'f_test')
    plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], color = 'tab:blue', label = 'pred')
    plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
                (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
                (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.3)
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

def plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A = None, model_name = 'gprg', num_group = None, grouping_method = 'random'):
    # counter plot, only for mean
    # refer https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py

    num_train = X_train.shape[0]

    fig, (ax1, ax2, ax3) = plt.subplots(figsize = (20,10), ncols=3)

    if model_name == 'gprg':
        info = model_name + '_train_' + str(num_train) + '_group_' + str(num_group) + '_' + str(grouping_method)
    elif model_name == 'gpr':
        info = model_name + '_train_' + str(num_train)
    else:
        print('unknown model name!')
        info = 'None'

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
    z_test_pred_var = Y_test_var.reshape(len(z_test),)
    z_test_pred_ucb = z_test_pred + 2 * np.sqrt(z_test_pred_var)
    z_test_pred_lcb = z_test_pred - 2 * np.sqrt(z_test_pred_var)

    # Create grid values first.
    xi = np.linspace(X_test_range_low, X_test_range_high, ngridx)
    yi = np.linspace(X_test_range_low, X_test_range_high, ngridy)
    Xi, Yi = np.meshgrid(xi, yi)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)

    # ------------------------------------------------------------
    # subplot 1: |z_test - z_test_pred|
    z_diff_mean = np.abs(z_test - z_test_pred)
    # print(z_test)
    # print(z_diff)
    interpolator1 = tri.LinearTriInterpolator(triang, z_diff_mean)
    zi1 = interpolator1(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
    
    # vmin = 0.
    # vmax = -0.5
    # levels = np.linspace(vmin, vmax, 14) 
    levels = 14

    ax1.contour(xi, yi, zi1, levels=levels, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi1, levels=levels, cmap="RdBu_r")
    ax1.plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    # if A is None:
    #     ax1.plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    # else:
    #     for i in range(A.shape[0]):
    #         ax1.plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1],'.',ms = 3)
    ax1.set_xlabel('X[:,0]')
    ax1.set_ylabel('X[:,1]')
    ax1.set_title(info + ' (pred mean - true mean)')
    plt.colorbar(cntr1, ax=ax1)

    # ------------------------------------------------------------
    # subplot 2: pred ucb - true mean
    
    z_diff_ucb = np.abs(z_test_pred_ucb - z_test)
    # print(z_test)
    # print(z_diff)
    interpolator2 = tri.LinearTriInterpolator(triang, z_diff_ucb)
    zi2 = interpolator2(Xi, Yi)

    # vmin = -1.
    # vmax = 1.
    # levels = np.linspace(vmin, vmax, 14) 

    ax2.contour(xi, yi, zi2, levels=levels, linewidths=0.5, colors='k')
    cntr2 = ax2.contourf(xi, yi, zi2, levels=levels, cmap="RdBu_r")
    ax2.plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    ax2.set_xlabel('X[:,0]')
    ax2.set_ylabel('X[:,1]')
    ax2.set_title(info + ' (pred ucb - true mean)')
    plt.colorbar(cntr2, ax=ax2)

    # ------------------------------------------------------------
    # subplot 3: true mean - pred lcb 
    
    z_diff_lcb = np.abs(z_test - z_test_pred_lcb)
    # print(z_test)
    # print(z_diff)
    interpolator3= tri.LinearTriInterpolator(triang, z_diff_lcb)
    zi3 = interpolator3(Xi, Yi)

    ax3.contour(xi, yi, zi3, levels=levels, linewidths=0.5, colors='k')
    cntr3 = ax3.contourf(xi, yi, zi3, levels=levels, cmap="RdBu_r")
    ax3.plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    ax3.set_xlabel('X[:,0]')
    ax3.set_ylabel('X[:,1]')
    ax3.set_title(info + ' (true mean - pred lcb)')
    plt.colorbar(cntr3, ax=ax3)


    # plt.contour(X_train[:,0], X_train[:,1], Y_train,label = 'train')
    # plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], label = 'test')
    # plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], label = 'pred')
    # plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
    #             (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
    #             (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.5)
    # plt.legend()
    
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.savefig(info + '_2d' + '.png')
    # plt.show()