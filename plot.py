import matplotlib
import pretty_errors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.cm import ScalarMappable as sm
import numpy as np

x_shift = 0

X_train_range_low = -3. + x_shift
X_train_range_high = 3. + x_shift
X_test_range_low = -3. + x_shift
X_test_range_high = 3. + x_shift

# X_train_range_low = -10. + x_shift
# X_train_range_high = 10. + x_shift
# X_test_range_low = -10. + x_shift
# X_test_range_high = 10.+ x_shift


def plot_1d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A, 
group_centers, group_test, group_test_pred, group_test_var, 
model_name = 'gprg', num_group = None, grouping_method = 'random', X_train_range_low = -3., X_train_range_high = 3., shift = 0):

    num_train = X_train.shape[0]

    sorted_train_idx = np.argsort(X_train, axis = 0).reshape(X_train.shape[0],)
    sorted_test_idx = np.argsort(X_test, axis = 0).reshape(X_test.shape[0],)

    # print('test label: ', Y_test[sorted_test_idx])
    # print('test pred: ', Y_test_pred[sorted_test_idx])
    # print('test std: ', np.sqrt(Y_test_var[sorted_test_idx]))

    # if A is None:
    if A is None:
        plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.', color = 'tab:green', label = 'train')
    else:
        # plt.plot(X_train[sorted_train_idx,:], Y_train[sorted_train_idx,:], '.', color = 'tab:green', label = 'train')
        A_sorted = A[:,sorted_train_idx]
        for i in range(A.shape[0]):
            plt.plot(X_train[sorted_train_idx,:][A_sorted[i,:] == 1,:], Y_train[sorted_train_idx,:][A_sorted[i,:] == 1,:], '.')
    # if A_ast.shape[0] == Y_test.shape[0]:
    plt.plot(X_test[sorted_test_idx,:], f_test[sorted_test_idx,:], color = 'tab:orange', label = 'f_test')
    plt.plot(X_test[sorted_test_idx,:], Y_test_pred[sorted_test_idx,:], color = 'tab:blue', label = 'pred')
    plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
                (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
                (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.3)
    # else:
    sorted_group_idx = np.argsort(group_centers, axis = 0).reshape(group_centers.shape[0],)
    plt.plot(group_centers[sorted_group_idx,:], group_test[sorted_group_idx,:], color = 'tab:orange', marker = 'o', label = 'group_test')
    plt.plot(group_centers[sorted_group_idx,:], group_test_pred[sorted_group_idx,:], color = 'tab:blue', marker = 'o', label = 'group pred')
    for c in group_centers:
        plt.axvline(x=c, linestyle = '--', color = 'black',alpha = 0.3)
        # plt.fill_between(X_test[sorted_test_idx,:].reshape(X_test.shape[0],), 
        #             (Y_test_pred[sorted_test_idx,:] - 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],),
        #             (Y_test_pred[sorted_test_idx,:] + 2 * np.sqrt(Y_test_var[sorted_test_idx,:])).reshape(X_test.shape[0],), alpha = 0.3)
    plt.legend()
    if model_name == 'gprg':
        info = model_name + '_train_' + str(num_train) + '_group_' + str(num_group) + '_' + str(grouping_method)
    elif model_name == 'gpr':
        info = model_name + '_train_' + str(num_train)
    else:
        print('unknown model name!')
        info = 'None'
    # plt.ylim(-1.7, 1.2)
    plt.title(info)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(info + '_1d' + '.png')

def plot_2d(X_train, X_test, f_train, Y_train, f_test, Y_test, Y_test_pred, Y_test_var, A, 
group_centers, group_test, group_test_pred, group_test_var, 
model_name = 'gprg', num_group = None, grouping_method = 'random'):
    # counter plot, only for mean
    # refer https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py

    num_train = X_train.shape[0]

    fig, axes = plt.subplots(figsize = (30,10), nrows = 2, ncols=3)

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

    axes[0,0].contour(xi, yi, zi1, levels=levels, linewidths=0.5, colors='k')
    cntr1 = axes[0,0].contourf(xi, yi, zi1, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[0,0].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[0,0].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    # if A is None:
    #     axes[0,0].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    # else:
    #     for i in range(A.shape[0]):
    #         axes[0,0].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1],'.',ms = 3)
    axes[0,0].set_xlabel('X[:,0]')
    axes[0,0].set_ylabel('X[:,1]')
    axes[0,0].set_title(info + ' (pred mean - true mean)')
    plt.colorbar(cntr1, ax=axes[0,0])

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

    axes[0,1].contour(xi, yi, zi2, levels=levels, linewidths=0.5, colors='k')
    cntr2 = axes[0,1].contourf(xi, yi, zi2, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[0,1].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[0,1].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    axes[0,1].set_xlabel('X[:,0]')
    axes[0,1].set_ylabel('X[:,1]')
    axes[0,1].set_title(info + ' (pred ucb - true mean)')
    plt.colorbar(cntr2, ax=axes[0,1])

    # ------------------------------------------------------------
    # subplot 3: true mean - pred lcb 
    
    z_diff_lcb = np.abs(z_test - z_test_pred_lcb)
    # print(z_test)
    # print(z_diff)
    interpolator3= tri.LinearTriInterpolator(triang, z_diff_lcb)
    zi3 = interpolator3(Xi, Yi)

    axes[0,2].contour(xi, yi, zi3, levels=levels, linewidths=0.5, colors='k')
    cntr3 = axes[0,2].contourf(xi, yi, zi3, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[0,2].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[0,2].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    axes[0,2].set_xlabel('X[:,0]')
    axes[0,2].set_ylabel('X[:,1]')
    axes[0,2].set_title(info + ' (true mean - pred lcb)')
    plt.colorbar(cntr3, ax=axes[0,2])

    # --------------------------------------------------------------
    # subplot 4: true mean

    interpolator4= tri.LinearTriInterpolator(triang, z_test)
    zi4 = interpolator4(Xi, Yi)

    axes[1,0].contour(xi, yi, zi4, levels=levels, linewidths=0.5, colors='k')
    cntr4 = axes[1,0].contourf(xi, yi, zi4, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[1,0].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[1,0].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    axes[1,0].set_xlabel('X[:,0]')
    axes[1,0].set_ylabel('X[:,1]')
    axes[1,0].set_title(info + ' true mean')
    plt.colorbar(cntr4, ax=axes[1,0])


    # --------------------------------------------------------------
    # subplot 5: pred mean

    interpolator5= tri.LinearTriInterpolator(triang, z_test_pred)
    zi5 = interpolator5(Xi, Yi)

    axes[1,1].contour(xi, yi, zi5, levels=levels, linewidths=0.5, colors='k')
    cntr5 = axes[1,1].contourf(xi, yi, zi5, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[1,1].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[1,1].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    axes[1,1].set_xlabel('X[:,0]')
    axes[1,1].set_ylabel('X[:,1]')
    axes[1,1].set_title(info + ' pred mean')
    plt.colorbar(cntr5, ax=axes[1,1])

    # --------------------------------------------------------------
    # subplot 6: pred var

    interpolator6= tri.LinearTriInterpolator(triang, z_test_pred_var)
    zi6 = interpolator6(Xi, Yi)

    axes[1,2].contour(xi, yi, zi6, levels=levels, linewidths=0.5, colors='k')
    cntr6 = axes[1,2].contourf(xi, yi, zi6, levels=levels, cmap="RdBu_r")
    if A is None:
        axes[1,2].plot(X_train[:,0], X_train[:,1],'ko',ms = 3)
    else:
        for i in range(A.shape[0]):
            axes[1,2].plot(X_train[A[i,:] == 1,0], X_train[A[i,:] == 1,1], 'o', ms = 3)
    axes[1,2].set_xlabel('X[:,0]')
    axes[1,2].set_ylabel('X[:,1]')
    axes[1,2].set_title(info + ' pred var')
    plt.colorbar(cntr6, ax=axes[1,2])


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