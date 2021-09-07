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

    for alg in ['GPOO', 'StoOO', 'Random', 'GPTree']:
    # for alg in ['GPOO']:
        saved_file = save_folder + alg + '_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'

        if os.path.isfile(saved_file):
            with open(saved_file, 'rb') as handle:
                data_dict = pickle.load(handle)
                regret_dict[alg + ' Center'] = data_dict['center']
                if alg != 'GPTree':
                    regret_dict[alg + ' Ave'] = data_dict['ave']
        else:
            print('Warning: ', str(saved_file) + ' not exist. Please check.')

    # fig, axes = plt.subplots(1, 1, figsize = (4,8))
    plot_name = 'Regret_' + str(n) + '_' + str(n_repeat)
    plot_regret_one(regret_dict, plot_name, budget = n, n_repeat=n_repeat, save_folder=save_folder)

    # raise Exception

# -------------------------------------------------------------------------------
# function and delta

# def f(x):
#     return (np.sin(13.0 * x) * np.sin(27.0*x) + 1)/2.0

def construct_model_f(opt_num, my_kernel):
    """Generate unknown f to be optimised by 
    posterior of a the known GP
    """
    size = 1

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

    # option 2:
    if opt_num == '2':
        sample_size = 9
        X = np.array([0.05, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.9, 0.95]).reshape(sample_size,1)
        Y = np.array([0.1, 0.3, 0.15, 0.35, 0.12, 0.85, 0.05, 0.98, 0.3]).reshape(sample_size,1)

    # option 3:
    if opt_num == '3':
        np.random.seed(2021)
        sample_size = 9
        X = []
        Y = []
        split_list = np.linspace(arms_range[0], arms_range[1], num = 9)
        for i in range(len(split_list)-1):
            center = (split_list[i] + split_list[i+1])/2.0
            y = np.random.uniform(0.0, 0.2)
            X.append(center)
            Y.append(y) # each center is assigned to a value 0~0.2

            another = np.random.uniform(split_list[i], split_list[i+1])
            X.append(another)
            Y.append(np.random.uniform(0.4, 0.6))

        X.append(0.9)
        Y.append(0.9)
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1,1)
        plt.scatter(X,Y)

    # option 4
    if opt_num == '4':
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

    # option 5
    if opt_num == '5':
        # construct tree with level hmax = 4, and put each node's representative point (center) to be low values. And run GPOO with the same hmax
        # grid splitting the arm range, and pick relatively low values for center and relatively high values for others points
        print(opt_num)
        tree_level = 4
        X = []
        Y = []

        for l in range(1, tree_level+2): # l: 1...tree_level
            split_list = np.linspace(arms_range[0], arms_range[1], num = l)
            for i in range(len(split_list)-1):
                center = (split_list[i] + split_list[i+1])/2.0
                print(center)
                X.append(center)
                Y.append(0.1)

                # if l == tree_level:
                    # another = center + 1.0 * (split_list[i+1] - split_list[i])/2
                    # X.append(another)
                    # Y.append(r.uniform(0.2, 0.25))
                    # Y.append(0.3)
                X.append(split_list[i])
                Y.append(0.3)
                X.append(split_list[i+1])
                Y.append(0.3)


        X.append(0.95)
        Y.append(0.5)
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1,1)

    if opt_num == '6':
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

    if opt_num == '7':
        # If you make the amplitudes increase from left to right, then your regret plot won't suddenly decrease like that.
        
        lengthscale = 0.02 # 0.05 # kernel para
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
            Y.append(center/2.0) # each center is assigned to a value 0~0.2

            # another = r.uniform(split_list[i], split_list[i+1])
            another = center + 2.0 * (split_list[i+1] - split_list[i])/3
            X.append(another)
            # Y.append(r.uniform(0.2, 0.25))
            Y.append((center + 0.1)/2.0)

        # X.append(0.94)
        # Y.append(0.5)
        # X.append(0.945)
        # Y.append(0.6)
        X.append(0.95)
        Y.append(0.8)
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
axes.scatter(X_samples,Y_samples)
plt.savefig(save_folder + 'posterior_f' + str(opt_num) + '.pdf', bbox_inches='tight')

# def opt_x(f, arms_range, f_type):
#     size = 1000
    
#     x = np.linspace(arms_range[0], arms_range[1], size)
#     if f_type == 'gp':
#         x = x.reshape(-1,1)
#         f_list = f(x)
#     else:
#         f_list = []
#         for i in x:
#             f_list.append(f(x))

#     # REVIEW: we assume there is an unique opt point
#     return np.argmax(f_list)
    
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

if 'DOO' in run_alg:
    # FIXME: not updated

    doo1 = DOO(f=f, delta = delta1, root_cell = arms_range, n = n, k=k, reward_type=reward_type)

    rec_node1 = doo1.rec()
    print(rec_node1.features)
    print(rec_node1.depth)
    # print([node.features for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    doo2 = DOO(f=f, delta = delta1, root_cell = arms_range, n = n, k=k, s=10, reward_type='ave')
    # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

    rec_node2 = doo2.rec()
    print(rec_node2.features)
    print(rec_node2.depth)
    # print(node.features for node in doo2.evaluated_nodes)
    # print(doo2.evaluated_fs)

    diff_node = []
    diff_f = []
    for i, node1 in enumerate(doo1.evaluated_nodes):
        for j, node2 in enumerate(doo2.evaluated_nodes):
            if node1.cell[0] == node2.cell[0] and node1.cell[1] == node2.cell[1]:
                # print('i: ', i)
                # print('j:', j)
                # assert node1.features - node2.features < 1e-3
                diff_node.append(node1.features)
                diff_f.append(np.abs(doo1.evaluated_fs[i] - doo2.evaluated_fs[j]))
                # diff_dict[node1.features] = 
                break
                

    print(len(doo1.evaluated_fs))
    print(len(doo2.evaluated_fs))

    # sanity check: Is the reward at the center the same as the average reward? Could be some scaling, but let's ignore that.
    # Scaling may not change the choice of arm, even if the rewards are different.

    plt.scatter(diff_node, diff_f, c = range(len(diff_node)))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('|center reward - ave reward|')
    plt.savefig('group_bandits/diff.pdf')

    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_two(arms_range, f, doo1, doo2, 'center v.s. ave')

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
    
    


    


# print('ave:')
# print('regret: ', regret_gpoo2)
# print(gpoo2.rec_node.features)
# print(gpoo2.rec_node.depth)

# print(node.features for node in doo2.evaluated_nodes)
# print(doo2.evaluated_fs)
        
# print(len(gpoo1.evaluated_fs))
# print(len(gpoo2.evaluated_fs))


# plot(arms_range, f, doo1, 'doo1')
# plot(arms_range, f, doo2, 'doo2')
# plot_regret_two(regret_gpoo1, regret_gpoo2, 'GPOO center v.s. ave')
# plot_two(arms_range, gpoo1.f, gpoo1, gpoo2, 'GPOO center v.s. ave')




# sto2 = StoOO(arms_range, f, delta1, k, n, reward_type, eta)

# rec_sto2 = sto2.rec()
# print(rec_sto2.features)
# print(rec_sto2.depth)

# sto3 = GPTree(arms_range, f, delta1, k, n, reward_type, eta)

# rec_sto3 = sto3.rec()
# print(rec_sto3.features)
# print(rec_sto3.depth)
        
# print(len(sto1.evaluated_fs))
# print(len(sto2.evaluated_fs))
# print(len(sto3.evaluated_fs))


# # plot(arms_range, f, doo1, 'doo1')
# # plot(arms_range, f, doo2, 'doo2')
# plot_three(arms_range, f, sto1, sto2, sto3, 'GPOO v.s. StoOO v.s. GPTree (center)')

    