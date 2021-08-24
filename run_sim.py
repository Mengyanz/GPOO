from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group
import pretty_errors
from DOO import * 

np.random.seed(2021)

# -------------------------------------------------------------------------------
# reproduce Fig 3.7

run_DOO = False
run_StoOO = True
run_GPStoOO = False

n = 100
k = 2
arms_range = [0.0, 1.0]
reward_type = 'center'
eta = 0.1

def f(x):
    return (np.sin(13.0 * x) * np.sin(27.0*x) + 1)/2.0

def delta1(h):
    return 14.0 * 2**(-h)

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


if run_GPStoOO:
    sto1 = GPStoOO(
        f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=1, reward_type = 'center', sigma = 0.1, eta=0.1
        )

    rec_sto1 = sto1.rec()
    print(rec_sto1.features)
    print(rec_sto1.depth)
    # print([node.features for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    sto2 = GPStoOO(
        f=f, delta=delta1, root_cell=arms_range, n=n, k=k, d=1, s=10, reward_type = 'ave', sigma = 0.1, eta=0.1
    )
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
    plot_two(arms_range, f, sto1, sto2, 'GPStoOO center v.s. ave')

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
    # plot_three(arms_range, f, sto1, sto2, sto3, 'GPStoOO v.s. StoOO v.s. GPTree (center)')

        