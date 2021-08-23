from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group
import pretty_errors

np.random.seed(2021)

class Tree():
    """
    Tree class. 

    Parameters
    -------------------------------------------------
    features: array
        s x d feature matrix of the current node
        where s is num of split (center: s = 1, ave: s > 1 is user specified), 
        d is the feature dimension
    center: array
        center node feature vector
        recorded for plotting
    cell: list (of cell members, or cell range)
        the cell range of a tree node
    children: list Tree instances
        the children node of the current node
        the children of leaf node is empty list. 
    depth: int
        the depth of the current node (root depth is 0)
    parent: Tree instance or None
        the parent node of the current node. 
        the root parent is None.
    range: 
    
    """
    def __init__(self) -> None:
        self.features = None
        self.center = None
        self.cell = None
        self.children = []
        self.depth = None
        self.parent = None

class Base():
    # TODO: 20210822 the current version only supports 1d interval inputs
    """
    Base class.

    Parameters
    ----------------------------------------------------------------------------
    f: function 
        the function to be optimised, take arm feature as argument
    delta: function
        upper bound of diameters, a func of h
    root_cell: list (of cell members, or cell range)
        root cell, i.e. all arms/feature space
    # FIXME: the current version of DOO treats n as expansion budget, evaluation budget is kn
    n: int
        sample budget 
    k: int
        k-ary tree, each node can have k children
    d: int 
        feature dimension
    s: int
        number of grid split for generating features
        if the reward type is center, s = 1
        if the reward type is ave: s > 1
    reward_type: string
        choices: 'center', 'ave'
    sigma: float
        noise standard deviation 
    eta: float
        error probability, [0,1]

    root: Tree instance
        root node
    leaves: list of Tree instances
        leaf nodes
    evaluated_nodes: list of Tree instances
        evaluated nodes
    evaluated_fs: list
        evaluated rewards (same order as evaluated_nodes)
    bvalues: dict
        key: Tree instance; value: latest bvalue of the key node
    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma=0.0, eta=0.0) -> None:
        
        self.f = f # the function to be optimised, take arm feature as argument
        self.delta = delta # upper bound of diameters, a func of h
        
        self.n = n # sample budget
        self.k = k # k-ary tree, each node can have k children
        self.d = d # feature dim
        self.s = s
        self.reward_type = reward_type
        self.sigma = sigma
        self.eta = eta 

        assert reward_type in {'center', 'ave'}
        if reward_type == 'center':
            assert self.s == 1
        if reward_type == 'ave':
            assert self.s > 1

        self.root = Tree()
        self.root.features = self.gene_feature(root_cell)
        self.root.center = self.center(root_cell)
        self.root.cell = root_cell 
        self.root.depth = 0 

        self.leaves = [self.root]

        self.evaluated_nodes = []
        self.evaluated_fs = []
        self.bvalues = {}


    def gene_feature(self, cell):
        # TODO: 20210822 the current version only supports 1d interval continuous inputs
        """
        Generate feature matrix of the tree node
        
        Parameter
        ----------------------------------------
        node: Tree instance

        Return 
        ----------------------------------------
        s x d array
        """
        # cell = node.cell
        features = []

        split_list = np.linspace(cell[0], cell[1], num = self.s + 1)
        for i in range(len(split_list)-1):
            features.append(self.center([split_list[i], split_list[i+1]]))

        return np.asarray(features)

    def center(self, interval):
        """ 
            Return the center of a set of arms
        """
        return (interval[0] + interval[1])/2.0

    def reward(self, node):
        rewards = []
        
        for feature in node.features:
            rewards.append(self.f(feature))

        noise = np.random.normal(0, self.sigma)

        return np.mean(rewards) + noise

    def expand(self, x):
        """
            Expand the select node into K children.
        """
        split_list = np.linspace(x.cell[0], x.cell[1], num = self.k + 1)
        for i in range(len(split_list)-1):
            node = Tree()
            node.cell = [split_list[i], split_list[i+1]]
            node.features = self.gene_feature(node.cell)
            node.center = self.center(node.cell)
            node.parent = x
            node.depth = x.depth + 1 
            x.children.append(node)

            self.leaves.append(node)


class DOO(Base):
    """Implementation of Deterministic Optimistic Optimisation algorithm 
    http://www.nowpublishers.com/articles/foundations-and-trends-in-machine-learning/MAL-038
    Fig 3.6 
    """
   
    def bvalue(self, x):
        """
            Return bvalue of tree node x
        """
        reward = self.reward(x)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)
        return reward + self.delta(x.depth)
    
    def rec(self):
        assert (self.sigma - 0.0) < 1e-5 # no noise

        sample_count = 0 
        while sample_count < self.n: # change n to sample budget
        # for i in range(self.n):
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    self.bvalues[x] = self.bvalue(x)
                    sample_count += 1
                    
            selected_node = max(self.bvalues, key = self.bvalues.get)
            del self.bvalues[selected_node]
            self.expand(selected_node)

        return self.evaluated_nodes[np.argmax(self.evaluated_fs)]

class StoOO(Base):
    """
    Implementation of Stochastic Optimistic Optimisation algorithm 
    http://www.nowpublishers.com/articles/foundations-and-trends-in-machine-learning/MAL-038
    Fig 3.9

    Parameters
    ----------------------------------------------------------
    T_dict: dict
        key: Tree instance 
        value: number of times the key node have been drawn
    samples: dict 
        key: Tree instance
        value: list of rewards observed at the key node
    deepest_expanded_node: Tree instance
        indicator of the deepest expanded node, 
        the center of which will be recommended after the budget is run out.
    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1, eta=0.1) -> None:
        super().__init__(f, delta, root_cell, n, k, d, s, reward_type, sigma, eta)

        self.T_dict = {} # key: node, value: number of times have been drawn
        self.samples = DefaultDict(list) # key: node, value: list of samples
        self.deepest_expanded_node = self.root

    def bvalue(self, x):
        """
            Return bvalue of tree node x
        """
        reward = self.reward(x)
        self.samples[x].append(reward)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)

        if len(self.samples[x]) >= 1:
            confidence_width = np.log(self.n**2/self.eta)/(2.0 * len(self.samples[x]))
        else: 
            confidence_width = np.inf
        return np.mean(self.samples[x]) + confidence_width + self.delta(x.depth)

    def rec(self):
        # self.bvalues[self.root] = self.bvalue(self.root)

        sample_count = 0 
        while sample_count < self.n: # change n to sample budget
        # for i in range(self.n):
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    # self.bvalues[x] = self.bvalue(x)
                    self.bvalues[x] = np.inf

            selected_node = max(self.bvalues, key = self.bvalues.get)
            if selected_node in self.T_dict.keys():
                self.T_dict[selected_node] += 1
            else:
                self.T_dict[selected_node] = 1
            self.bvalues[selected_node] = self.bvalue(selected_node)
            sample_count +=1
            
            thereshold = np.log(self.n**2/self.eta)/(2.0 * len(self.samples[selected_node]))
            # print('round ' +  str(i) + ' threshold ' + str(thereshold))
            if self.T_dict[selected_node] >= thereshold:
                del self.bvalues[selected_node]
                # FIXME: need to fix the case where there is more than one nodes in the deepest depth
                if selected_node.depth > self.deepest_expanded_node.depth:
                    self.deepest_expanded_node = selected_node
                self.expand(selected_node)

        return self.deepest_expanded_node


# ------------------------------------------------------------------------------
# Plot func 

def plot_tree(node, ax):
    ax.scatter(node.center, -node.depth, s=1.5, c = 'b')
    if len(node.children) > 0:
        for child in node.children:
            ax.plot([node.center, child.center], [-node.depth, - child.depth], c = 'gray', alpha = 0.5)
            plot_tree(child, ax)


def plot(arms_range, f, doo, axes):
    # fig, axes = plt.subplots(2, 1, figsize = (6,8), sharex=True)

    # data = []
    # neg_depth = []
    # for node in doo.evaluated_nodes:
    #     data.append(node.features)
    #     neg_depth.append(- node.depth)
    # axes[0].scatter(data, neg_depth, s = 1)

    plot_tree(doo.root, axes[0])
    
    x = np.linspace(arms_range[0], arms_range[1], 1000)
    axes[1].plot(x, f(x), c = 'r', alpha = 0.5)
    # plt.show()
    

def plot_two(arms_range, f, doo1, doo2, name = 'center'):
    fig, axes = plt.subplots(2, 2, figsize = (12,8), sharex=True)
    plot(arms_range, f, doo1, axes[:, 0])
    plot(arms_range, f, doo2, axes[:, 1])
    fig.suptitle(name)
    plt.savefig(name + '_doo.png')

def plot_three(arms_range, f, doo1, doo2, doo3, name = 'center'):
    fig, axes = plt.subplots(2, 3, figsize = (12,8), sharex=True)
    plot(arms_range, f, doo1, axes[:, 0])
    plot(arms_range, f, doo2, axes[:, 1])
    plot(arms_range, f, doo3, axes[:, 2])
    fig.suptitle(name)
    plt.savefig(name + '_doo.png')

            
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
    sto1 = GPStoOO(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto1 = sto1.rec()
    print(rec_sto1.features)
    print(rec_sto1.depth)
    # print([node.features for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    sto2 = StoOO(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto2 = sto2.rec()
    print(rec_sto2.features)
    print(rec_sto2.depth)

    sto3 = GPTree(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto3 = sto3.rec()
    print(rec_sto3.features)
    print(rec_sto3.depth)
            
    print(len(sto1.evaluated_fs))
    print(len(sto2.evaluated_fs))
    print(len(sto3.evaluated_fs))


    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_three(arms_range, f, sto1, sto2, sto3, 'GPStoOO v.s. StoOO v.s. GPTree (center)')

        