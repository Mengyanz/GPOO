from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group
import pretty_errors
from simulation import * 
import pretty_errors

# np.random.seed(2021)

# import pickle
# with open('save_regret.pickle', 'rb') as handle:
#     data_dict = pickle.load(handle)
# regret_gpoo1 = data_dict['center']
# regret_gpoo2 = data_dict['ave']
# plot_regret_two(regret_gpoo1, regret_gpoo2, 'GPOO center v.s. ave', n_repeat=len(regret_gpoo1))

# raise Exception

# -------------------------------------------------------------------------------
# reproduce Fig 3.7

run_DOO = False
run_StoOO = False
run_GPOO = True

n = 80
n_repreat = 200
k = 2
arms_range = [0.0, 1.0]
reward_type = 'center'
eta = 0.1

def f(x):
    return (np.sin(13.0 * x) * np.sin(27.0*x) + 1)/2.0

def opt_x(f, arms_range, f_type):
    size = 1000
    
    x = np.linspace(arms_range[0], arms_range[1], size)
    if f_type == 'gp':
        x = x.reshape(-1,1)
        f_list = f(x)
    else:
        f_list = []
        for i in x:
            f_list.append(f(x))

    # REVIEW: we assume there is an unique opt point
    return np.argmax(f_list)
    
def delta1(h):
    # return 14.0 * 2**(-h)
    return 14 * 2**(-h)

def delta2(h):
    return 222.0 * 2**(-2.0 * h)

if run_DOO:

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
    sto1 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1)

    rec_sto1 = sto1.rec()
    print(rec_sto1.features)
    print(rec_sto1.depth)
    # print([node.features for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    sto2 = StoOO(f=f, delta=delta1, root_cell=arms_range, n=n, k=2, d=1, s=2, reward_type = 'ave', sigma = 0.1)
    # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

    rec_sto2 = sto2.rec()
    print(rec_sto2.features)
    print(rec_sto2.depth)
    # print(node.features for node in doo2.evaluated_nodes)
    # print(doo2.evaluated_fs)
            
    print(len(sto1.evaluated_fs))
    print(len(sto2.evaluated_fs))


    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_two(arms_range, f, sto1, sto2, 'StoOO center v.s. ave')

if run_GPOO:
    rep_regret_list1 = []
    rep_regret_list2 = []

    for i in range(n_repreat):
        print(i)

        gpoo1 = GPOO(
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, eta=0.1
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
            f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1, eta=0.1
        )
        regret_gpoo2 = gpoo2.rec()
        rep_regret_list2.append(regret_gpoo2)

    import pickle 
    data_dict = {}
    data_dict['center'] = rep_regret_list1
    data_dict['ave'] = rep_regret_list2
    
    with open('save_regret.pickle', 'wb') as handle:
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

        