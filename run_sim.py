from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group
from simulation import * 
import pickle
import pretty_errors

#-----------------------------------------------------------------------------------
# parameters

# np.random.seed(2021)
n = 80 # budget
n_repeat = 1 # number of repeat
arms_range = [0.0, 1.0] # root cell
reward_type = 'center' # 'center' or 'ave'
sigma = 0.1 # noise for observation (normal std)
k = 2 # k-ary tree
s = 10 # split for aggregated feedback
d = 1 # feature dim

lengthscale = 0.1 # kernel para
kernel_var = 0.5 # kernel para
gp_noise_var = kernel_var * 0.05 # 1e-10 # gp para
opt_flag = False # whether optimise parameters in gpr

run_DOO = False
run_StoOO = True
run_GPOO = False
run_GPTree = False
run_random = False
plot_regret = False
plot_tree = True

# ----------------------------------------------------------------------------------
# plot regret from saved file

if plot_regret:
    regret_dict = {}
    with open('gpoo_save_regret_80_200.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
        regret_dict['GPOO Center'] = data_dict['center']
        regret_dict['GPOO Ave'] = data_dict['ave']

    with open('gptree_save_regret.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
        regret_dict['GPTree Center'] = data_dict['center']
        regret_dict['GPTree Ave'] = data_dict['ave']

    with open('random_save_regret.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
        regret_dict['Random Center'] = data_dict['center']
        regret_dict['Random Ave'] = data_dict['ave']

    # fig, axes = plt.subplots(1, 1, figsize = (4,8))
    plot_regret_two(regret_dict, regret_dict, 'Regret GPTree', budget = n, n_repeat=n_repeat)

    raise Exception

# -------------------------------------------------------------------------------
# function and delta

# def f(x):
#     return (np.sin(13.0 * x) * np.sin(27.0*x) + 1)/2.0

def f(x):
    """Generate unknown f to be optimised by 
    posterior of a the known GP
    """
    sample_size = 5
    size = 1

    # X = np.random.uniform(0, 1., (sample_size, 1))
    # Y = np.sin(X) + np.random.randn(sample_size, 1)*0.05

    # option 1:
    X = np.array([0.05, 0.2, 0.4, 0.65, 0.9]).reshape(sample_size,1)
    # Y = np.array([0.9, 0.1, 0.95, 0.05, 0.98]).reshape(sample_size,1)
    Y = np.array([0.85, 0.1, 0.87, 0.05, 0.98]).reshape(sample_size,1)

    # option 2:

    kernel = GPy.kern.RBF(input_dim=d, 
                        variance=kernel_var, 
                        lengthscale=lengthscale)
    model = GPy.models.GPRegression(X,Y,kernel, noise_var=gp_noise_var)

    # return partial(model.posterior_samples_f, size=size)
    def wraps(*args):
        return model.predict(*args)[0]
    return wraps(x)

def get_opt_x(f, arms_range):
    """Empirical opt x.
    """
    size = 1000
    
    x = np.linspace(arms_range[0], arms_range[1], size).reshape(-1,1)
    f_list = f(x)
    
    # REVIEW: we assume there is an unique opt point
    return x[np.argmax(f_list)].reshape(-1,1)

# # test f
# testX = np.linspace(arms_range[0], arms_range[1], 100).reshape(-1, 1)
# posteriorTestY = f(testX)
# opt_x = get_opt_x(f, arms_range)

# for i in range(100):
#     plt.plot(testX, posteriorTestY)
# plt.scatter(opt_x, f(opt_x), c = 'red')
# plt.savefig('posterior_f.png')

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

if run_DOO:
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
    plt.savefig('group_bandits/diff.png')

    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_two(arms_range, f, doo1, doo2, 'center v.s. ave')

if run_StoOO:
    eta = 0.1
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):
        regret_list1 = []
        regret_list2 = []
        print('repeat: ', i)
        for b in range(n):
            print('budget: ', b)
            sto1 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=b, k=k, d=d, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x, eta = eta)

            regret_sto1 = sto1.rec()
            regret_list1.append(regret_sto1)

            sto2 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=b, k=k, d=d, s=s, reward_type = 'ave', sigma = 0.1, opt_x = opt_x, eta = eta)
            # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

            regret_sto2 = sto2.rec()
            regret_list2.append(regret_sto2)
        print('**************************************')

        if plot_tree:
            plot_two(arms_range, f, sto1, sto2, 'StoOO center v.s. ave')

    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    save_name = 'stooo _save_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if run_GPOO:
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):

        gpoo1 = GPOO(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = opt_x,
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
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1, opt_x = get_opt_x(f, arms_range),
            lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
        )
        regret_gpoo2 = gpoo2.rec()
        rep_regret_list2.append(regret_gpoo2)

    if plot_tree:
        plot_two(arms_range, gpoo1.f, gpoo1, gpoo2, 'GPOO center v.s. ave')

    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    save_name = 'gpoo_save_regret_' + str(n) + '_' + str(n_repeat) + '.pickle'
    with open(save_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if run_GPTree:
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repeat):
        print(i)

        gptree1 = GPTree(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, 
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

        gptree2 = GPTree(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1,
            lengthscale = lengthscale, kernel_var = kernel_var, gp_noise_var = gp_noise_var, opt_flag = opt_flag
        )
        regret_gptree2 = gptree2.rec()
        rep_regret_list2.append(regret_gptree2)

    plot_two(arms_range, gptree1.f, gptree1, gptree2, 'GPTree center v.s. ave')

    import pickle 
    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    with open('gptree_save_regret.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if run_random:
    
    rep_regret_list1 = []
    rep_regret_list2 = []
    for i in range(n_repeat):
        regret_list1 = []
        regret_list2 = []
        for b in np.linspace(1, n, n):
            b = int(b)
            ran1 = Random(
                f=f, delta=delta1, root_cell=arms_range, n=b, k=b, d=1, s=1, reward_type = 'center', sigma = 0.1, 
            )
            ran2 = Random(
                f=f, delta=delta1, root_cell=arms_range, n=b, k=b, d=1, s=10, reward_type = 'ave', sigma = 0.1, 
            )
            regret_list1.append(ran1.rec())
            regret_list2.append(ran2.rec())
        rep_regret_list1.append(regret_list1)
        rep_regret_list2.append(regret_list2)

    # plot_two(arms_range, ran1.f, ran1, ran2, 'Random center v.s. ave')

    import pickle 
    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    with open('random_save_regret.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

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

    