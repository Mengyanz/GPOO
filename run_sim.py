from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group
from simulation import * 
import pickle
import os
import argparse
import sys
import pandas as pd 
import xarray as xr
import pretty_errors

plt.style.use('double_col.mplstyle')

#-----------------------------------------------------------------------------------
# parameters

parser = argparse.ArgumentParser(description='Run Simulation for GPOO project.')
parser.add_argument('opt_num', type = int, help = 'choose what f to use, choices: 1,2,3')
parser.add_argument('--n', type = int, help = 'budget (should be positive integer)')
parser.add_argument('--r', type = int, help = 'number of repeat (should be positive integer)')
parser.add_argument('--alg', nargs='*', help = 'please list all algorithms to run. Choices: StoOO, GPOO, GPTree, Random, SK')
# args = parser.parse_args()
args,_ = parser.parse_known_args()

opt_num = str(args.opt_num) 
save_folder = 'GPOO_results' + opt_num + '/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# np.random.seed(2021)
if args.n is not None:
    n = args.n  
else:
    n = 80 # budget
if args.r is not None:
    n_repeat = args.r
else:
    n_repeat = 100 # number of repeat
if args.alg is not None:
    run_alg = args.alg
    print(run_alg)
else:
    run_alg = ['StoOO', 'GPOO', 'GPTree', 'Random']

hmax = 10
arms_range = [0.0, 1.0] # root cell
reward_type = 'center' # 'center' or 'ave'
sigma = 0.1 # noise for observation (normal std)
k = 2 # k-ary tree
s = 10 # split for aggregated feedback
d = 1 # feature dim


lengthscale = 0.05 # kernel para
kernel_var = 0.1 # kernel para
gp_noise_var = kernel_var * 0.05 # 1e-10 # gp para

# my_kernel = GPy.kern.StdPeriodic(
#     input_dim=d, 
#     # variance=kernel_var, 
#     lengthscale=0.2, 
#     variance = 0.5,
#     period=1
#     # ,n_freq=10,lower=0.0, upper=0.3,active_dims=0, name=None
# )

my_kernel = GPy.kern.RBF(input_dim=d, 
                    variance=kernel_var, 
                    lengthscale=lengthscale)
opt_flag = False # whether optimise parameters in gpr
plot_regret = True
plot_tree = True

# ----------------------------------------------------------------------------------
# plot regret from saved file

if plot_regret:
    regret_dict = {}
    regret_ave_dict = {}
    regret_center_dict = {}

    # for alg in ['GPOO', 'StoOO', 'Random', 'GPTree'']:
    for alg in ['GPOO', 'StoOO', 'GPTree']:
        saved_file = save_folder + alg + '_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'

        if os.path.isfile(saved_file):
            with open(saved_file, 'rb') as handle:
                data_dict = pickle.load(handle)
                regret_dict[alg + ' S=1'] = data_dict['center']
                regret_center_dict[alg] = data_dict['center']
                if alg != 'GPTree':
                    regret_dict[alg + ' S=10'] = data_dict['ave']
                    regret_ave_dict[alg] = data_dict['ave']
        else:
            print('Warning: ', str(saved_file) + ' not exist. Please check.')

    # fig, axes = plt.subplots(1, 1, figsize = (4,8))
    plot_name = 'Regret_' + str(n) + '_' + str(n_repeat)
    plot_regret_one(regret_dict, plot_name, budget = n, n_repeat=n_repeat, save_folder=save_folder)

    # plot_name = 'Regret_' + str(n) + '_' + str(n_repeat) + '_center'
    # plot_regret_one(regret_center_dict, plot_name, budget = n, n_repeat=n_repeat, save_folder=save_folder, plot_title='S = 1')

    # plot_name = 'Regret_' + str(n) + '_' + str(n_repeat) + '_ave'
    # plot_regret_one(regret_ave_dict, plot_name, budget = n, n_repeat=n_repeat, save_folder=save_folder, plot_title='S = 10')

# -------------------------------------------------------------------------------
# function and delta

# def f(x):
#     return (np.sin(13.0 * x) * np.sin(27.0*x) + 1)/2.0

def construct_model_f(opt_num, my_kernel):
    """Generate unknown f to be optimised by 
    posterior of a the known GP
    """

    # X = np.random.uniform(0, 1., (sample_size, 1))
    # Y = np.sin(X) + np.random.randn(sample_size, 1)*0.05

    # option 1:
    if opt_num == '1':
        sample_size = 5
        X = np.array([0.05, 0.2, 0.4, 0.65, 0.9]).reshape(sample_size,1)
        Y = np.array([0.85, 0.1, 0.87, 0.05, 0.98]).reshape(sample_size,1)

        my_kernel = GPy.kern.RBF(input_dim=d, 
                    variance=kernel_var, 
                    lengthscale=0.1)

    # option 4
    if opt_num == '2':
        # create a function that has different frequency on the left and right? I.e. wiggly on the left, very low wiggle on the right. E.g. only one wave for the high amplitude, but 10 waves for the low amplitude.

        r = np.random.RandomState(2021)
        X = []
        Y = []
        split_list = np.linspace(arms_range[0], 0.9, num = 10)
        for i in range(len(split_list)-1):
            center = (split_list[i] + split_list[i+1])/2.0
            # y = r.uniform(0.05, 0.1)
            y = 0.1
            X.append(center)
            Y.append(y) # each center is assigned to a value 0~0.2

            # another = r.uniform(split_list[i], split_list[i+1])
            another = center + 2.0 * (split_list[i+1] - split_list[i])/3
            X.append(another)
            # Y.append(r.uniform(0.2, 0.25))
            Y.append(0.2)

        X.append(0.95)
        Y.append(0.9)
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1,1)

    # option 6
    if opt_num == '3':
        # create a function that has different frequency on the left and right? I.e. wiggly on the left, very low wiggle on the right. E.g. only one wave for the high amplitude, but 10 waves for the low amplitude.
        
        lengthscale = 0.01 # 0.05 # kernel para
        my_kernel = GPy.kern.RBF(input_dim=d, 
                    variance=kernel_var, 
                    lengthscale=lengthscale)

        r = np.random.RandomState(2021)
        sample_size = 9
        X = []
        Y = []
        split_list = np.linspace(arms_range[0], 0.9, num = 10)
        for i in range(len(split_list)-1):
            center = (split_list[i] + split_list[i+1])/2.0
            # y = r.uniform(0.05, 0.1)
            y = 0.1
            X.append(center)
            Y.append(y) # each center is assigned to a value 0~0.2

            # another = r.uniform(split_list[i], split_list[i+1])
            another = center + 2.0 * (split_list[i+1] - split_list[i])/3
            X.append(another)
            # Y.append(r.uniform(0.2, 0.25))
            Y.append(0.2)

        X.append(0.94)
        Y.append(0.1)
        X.append(0.945)
        Y.append(0.2)
        X.append(0.95)
        Y.append(0.9)
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1,1)

    model = GPy.models.GPRegression(X,Y,my_kernel, noise_var=gp_noise_var)

    return model, X,Y

f_model,X_samples, Y_samples = construct_model_f(opt_num, my_kernel)

# return partial(model.posterior_samples_f, size=size)
def f(x):
    return f_model.predict(x)[0]

def get_opt_x(f, arms_range):
    """Empirical opt x.
    """
    size = 1000
    
    x = np.linspace(arms_range[0], arms_range[1], size).reshape(-1,1)
    f_list = f(x)
    
    # REVIEW: we assume there is an unique opt point
    return x[np.argmax(f_list)].reshape(-1,1)

# test f
testX = np.linspace(arms_range[0], arms_range[1], 1000).reshape(-1, 1)
posteriorTestY = f(testX)
opt_x = get_opt_x(f, arms_range)

fig, axes = plt.subplots(1, 1, figsize = (6,6))
# for i in range(1000):
axes.plot(testX, posteriorTestY, c = 'black')
axes.scatter(opt_x, f(opt_x), c = 'red')
axes.set_xlabel('Arms')
axes.set_ylabel('Reward Function')
axes.set_ylim(-0.05,1)
# axes.scatter(X_samples,Y_samples)
plt.savefig(save_folder + 'posterior_f' + str(opt_num) + '.pdf', bbox_inches='tight')

    
def delta1(h):
    # return 14.0 * 2**(-h)
    return 14 * 2**(-h)

def delta2(h):
    return 222.0 * 2**(-2.0 * h)

#------------------------------------------------------------------------------------
# run algorithm

opt_x = get_opt_x(f, arms_range)
print('opt_x: ', opt_x)
print('opt f: ', f(opt_x))

if 'StoOO' in run_alg:
    eta = 0.1
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):
        regret_list1 = []
        regret_list2 = []
        print('repeat: ', i)
        for b in range(n):
            # print('budget: ', b)
            sto1 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=b, k=k, d=d, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x, eta = eta)

            regret_sto1 = sto1.rec()
            regret_list1.append(regret_sto1)

            sto2 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=b, k=k, d=d, s=s, reward_type = 'ave', sigma = 0.1, opt_x = opt_x, eta = eta)
            # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

            regret_sto2 = sto2.rec()
            regret_list2.append(regret_sto2)
        # print('**************************************')

        rep_regret_list1.append(regret_list1)
        rep_regret_list2.append(regret_list2)

    if plot_tree:
        plot_two(arms_range, f, sto1, sto2, 'StoOO', save_folder=save_folder)

    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    save_name = save_folder + 'StoOO_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)      

if 'GPOO' in run_alg:
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):
        print('repeat: ', i)
        gpoo1 = GPOO(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x, hmax = hmax,
            lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
            )
        regret_gpoo1 = gpoo1.rec()
        rep_regret_list1.append(regret_gpoo1)

        # print('center:')
        # print('regret: ', regret_gpoo1)
        # print(gpoo1.rec_node.features)
        # print(gpoo1.rec_node.depth)
        # print('*****************************')

        # print([node.features for node in doo1.evaluated_nodes])
        # print(doo1.evaluated_fs)
        # print()

        gpoo2 = GPOO(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1, opt_x = get_opt_x(f, arms_range), hmax = hmax,
            lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
        )
        regret_gpoo2 = gpoo2.rec()
        rep_regret_list2.append(regret_gpoo2)

    if plot_tree:
        plot_two(arms_range, gpoo1.f, gpoo1, gpoo2, 'GPOO',save_folder=save_folder)

    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    save_name = save_folder + 'GPOO_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if 'GPTree' in run_alg:
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):
        print('repeat: ', i)

        gptree1 = GPTree(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x,
            lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
            )
        regret_gptree1 = gptree1.rec()
        rep_regret_list1.append(regret_gptree1)

        # print('center:')
        # print('regret: ', regret_gptree1)
        # print(gptree1.rec_node.features)
        # print(gptree1.rec_node.depth)
        # print('*****************************')

        # print([node.features for node in doo1.evaluated_nodes])
        # print(doo1.evaluated_fs)
        # print()

        # NOTE: not running ave case for GPTree
        # gptree2 = GPTree(
        #     f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1, opt_x = opt_x,
        #     lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
        # )
        # regret_gptree2 = gptree2.rec()
        # rep_regret_list2.append(regret_gptree2)

    if plot_tree:
        plot_two(arms_range, gptree1.f, gptree1, gptree1, 'GPTree Only Center', save_folder=save_folder)

    import pickle 
    data_dict = {}
    data_dict['center'] = rep_regret_list1
    # data_dict['ave'] = rep_regret_list2
    
    save_name = save_folder+ 'GPTree_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if 'Random' in run_alg:
    
    rep_regret_list1 = []
    rep_regret_list2 = []
    for i in range(n_repeat):
        print('repeat: ', i)
        regret_list1 = []
        regret_list2 = []
        for b in np.linspace(1, n, n):
            b = int(b)
            ran1 = Random(
                f=f, delta=delta1, root_cell=arms_range, n=b, k=b, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x
            )
            ran2 = Random(
                f=f, delta=delta1, root_cell=arms_range, n=b, k=b, d=1, s=10, reward_type = 'ave', sigma = 0.1, opt_x = opt_x
            )
            regret_list1.append(ran1.rec())
            regret_list2.append(ran2.rec())
        rep_regret_list1.append(regret_list1)
        rep_regret_list2.append(regret_list2)

    if plot_tree:
        plot_two(arms_range, ran1.f, ran1, ran2, 'Random', save_folder=save_folder)

    import pickle 
    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    save_name = save_folder + 'Random_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if 'SK' in run_alg:
    # s v.s. k v.s. regret
    saved_file = save_folder + 'SK_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    s_list = [1, 10, 20, 30, 40, 50]
    k_list = [2,3,4,5]
    if os.path.isfile(saved_file):
        with open(saved_file, 'rb') as handle:
            regret_dict = pickle.load(handle)
        df = pd.DataFrame.from_dict(regret_dict, orient = 'index', columns = k_list)
        df.index.name = 'S'
        df.columns.name = 'K'
    
        xr_data = xr.DataArray(np.log(df))
        xr_data.plot()
        plt.title('N='+str(n))
        # plt.clim(-2.25,-0.05)
        plt.xticks(k_list)
        plt.yticks(s_list)
        plt.savefig(save_folder + 'SK_regret_' + str(n) + '.pdf')
    else:
        # regret_dict = {}
        regret_dict = DefaultDict(list) # only record mean 
        regret_whole_dict = {}
        
        for s in s_list:
            for k in k_list:
                print('s, k: ' +  str(s) + ' ' + str(k))
                if s == 1:
                    reward_type = 'center'
                else:
                    reward_type = 'ave'
                regret_list = []
                for i in range(n_repeat):
                    print('repeat: ', i)
                    gpoo = GPOO(
                        f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=d, s=s, reward_type = reward_type, sigma = sigma, opt_x = get_opt_x(f, arms_range), hmax = hmax,
                        lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
                    )
                    regret_gpoo = gpoo.rec()[-1]
                    regret_list.append(regret_gpoo)
                key = str(s) + ',' + str(k)
                regret_whole_dict[key] = regret_list 
                regret_dict[s].append(np.mean(regret_list))
        
        
        with open(saved_file, 'wb') as handle:
            pickle.dump(regret_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        saved_whole_file = save_folder + 'SK_whole_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
        with open(saved_whole_file, 'wb') as handle:
            pickle.dump(regret_whole_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    