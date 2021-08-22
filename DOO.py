from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import GPy
from numpy.lib.function_base import select
from gpr_group_model import GPRegression_Group

np.random.seed(2021)

class Tree():
    def __init__(self) -> None:
        self.data = None
        self.children = []
        self.depth = None
        self.parent = None
        self.range = []


class DOO():
    """Implementation of DEterministic Optimistic Optimisation algorithm 
    http://www.nowpublishers.com/articles/foundations-and-trends-in-machine-learning/MAL-038
    Fig 3.6 
    """
    def __init__(self, arms_range, f, delta, k, n, reward_type = 'center', d = 1) -> None:
        self.d = d # feature dim
        self.arms = arms_range # 1d continuous arm feature range
        self.f = f # the function to be optimised, take arm feature as argument
        self.delta = delta # upper bound of diameters, a func of h
        self.k = k # k-ary tree, each node can have k children
        self.n = n # budget of expansion, evaluation budget is kn
        self.reward_type = reward_type

        self.root = Tree()
        self.root.data = self.center(self.arms)
        self.root.range = self.arms
        self.root.depth = 0 

        self.leaves = [self.root]

        self.evaluated_nodes = []
        self.evaluated_fs = []
        self.bvalues = {}


    def center(self, arms):
        """ 
            Return the center of a set of arms
        """
        return (arms[0] + arms[1])/2.0

    def reward(self, x):
        if self.reward_type == 'center':
            return self.f(x.data)
        elif self.reward_type == 'ave':
            # print('ave')
            return np.mean(self.f(np.linspace(x.range[0], x.range[1], 10)))


    def bvalue(self, x):
        """
            Return bvalue of tree node x
        """
        reward = self.reward(x)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)
        return reward + self.delta(x.depth)

    def expand(self, x):
        """
            Expand the select node x into K children.
        """
        split_list = np.linspace(x.range[0], x.range[1], num = self.k + 1)
        for i in range(len(split_list)-1):
            node = Tree()
            node.range = [split_list[i], split_list[i+1]]
            node.data = self.center(node.range)
            node.parent = x
            node.depth = x.depth + 1 
            x.children.append(node)

            self.leaves.append(node)

    def rec(self):
        for i in range(self.n):
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    self.bvalues[x] = self.bvalue(x)
                    
            selected_node = max(self.bvalues, key = self.bvalues.get)
            del self.bvalues[selected_node]
            self.expand(selected_node)

        return self.evaluated_nodes[np.argmax(self.evaluated_fs)]

class StoOO(DOO):
    def __init__(self, arms_range, f, delta, k, n, reward_type = 'center', d = 1, eta = 0.1) -> None:
        super().__init__(arms_range, f, delta, k, n, reward_type=reward_type, d)
        self.eta = eta # error probability

        self.T_dict = {} # key: node, value: number of times have been drawn
        self.samples = DefaultDict(list) # key: node, value: list of samples
        self.deepest_expanded_node = self.root

    def reward(self, x):
        noise = self.eta * np.random.uniform(0,1)
        if self.reward_type == 'center':
            return self.f(x.data) + noise
        elif self.reward_type == 'ave':
            # print('ave')
            return np.mean(self.f(np.linspace(x.range[0], x.range[1], 10))) + noise

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
        for i in range(self.n):
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
            
            thereshold = np.log(self.n**2/self.eta)/(2.0 * len(self.samples[selected_node]))
            # print('round ' +  str(i) + ' threshold ' + str(thereshold))
            if self.T_dict[selected_node] >= thereshold:
                del self.bvalues[selected_node]
                if selected_node.depth > self.deepest_expanded_node.depth:
                    self.deepest_expanded_node = selected_node
                self.expand(selected_node)

        return self.deepest_expanded_node


class GPStoOO(StoOO):
    def __init__(self, arms_range, f, delta, k, n, reward_type = 'center', d = 1, eta = 0.1) -> None:
        super().__init__(arms_range, f, delta, k, n, reward_type, d, eta)
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        self.reward_type = reward_type 
        
        root_feature = np.array([self.root.data]).reshape(1,1) # TODO: might need to extend to multi-dim
        root_reward = np.array([self.reward(self.root)]).reshape(1,1)
        self.m = GPRegression_Group(root_feature, root_reward, self.kernel)
        self.bvalues[self.root] = self.bvalue(self.root, 0, reward=root_reward)

    def set_A(self)
        if self.reward_type == 'center':
            A = np.identity(self.n)
        elif self.reward_type == 'ave':
            A =  np.zeros((self.n, self.n * self.k))
        else:
            raise Exception("Invalid reward type. Only center or ave is accepted.")

    def update_posterior(self, x, y):
        x = np.array([x]).reshape(1,1)
        y = np.array([y]).reshape(1,1)
        self.m.set_XY(x, y)

    def beta(self,t):
        # TODO: need to change (based on theo analysis)
        # return 0.5 * np.log(t)
        # return 2 * np.log(np.pi**2 * t**2/(6 * self.eta))
        return 1

    def bvalue(self, x, t, reward = None):
        """
            Return bvalue of tree node x
        """
        if reward == None:
            reward = self.reward(x)
        self.samples[x].append(reward)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)

        self.update_posterior(x.data, reward)

        mu, var = self.m.predict(np.array([x.data]).reshape(1,1))

        return mu + self.beta(t) * np.sqrt(var) + self.delta(x.depth), self.beta(t) * np.sqrt(var)

    def rec(self):
        # self.bvalues[self.root] = self.bvalue(self.root)
        for i in range(self.n):
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    # self.bvalues[x] = self.bvalue(x)
                    self.bvalues[x] = np.inf

            selected_node = max(self.bvalues, key = self.bvalues.get)
            if selected_node in self.T_dict.keys():
                self.T_dict[selected_node] += 1
            else:
                self.T_dict[selected_node] = 1
            self.bvalues[selected_node], thereshold = self.bvalue(selected_node, i)
            
            # print('round ' +  str(i) + ' threshold ' + str(thereshold))
            # if self.delta(selected_node.depth) >= thereshold:
            if self.T_dict[selected_node] >= thereshold:
                del self.bvalues[selected_node]
                if selected_node.depth > self.deepest_expanded_node.depth:
                    self.deepest_expanded_node = selected_node
                self.expand(selected_node)

        return self.deepest_expanded_node

class GPTree(GPStoOO):
    """
        Algorithm in Shekhar et al. 2018
    """
    def __init__(self, arms_range, f, delta, k, n, reward_type = 'center', eta = 0.1, 
                alpha = 0.5, rho = 0.5, u = 2.0, v1 = 1.0, v2 = 1.0, C3 = 1.0, C2 = 1.0, D1=1) -> None:
        """
        alpha, rho (0,1)
        u >0
        0<v2<=1<=v1 
        C2,C3 > 0 (corollary 1)
        D1 >= 0 metric dimension (Defi 2)
        """
        # TODO: might need to change constant rate
        self.beta_n = 0.1 * np.sqrt(np.log(n) + u)
        self.betastd = {}

        # Todo: the following parameters might need to be chosen more carefully
        self.hmax = np.log(n) * (1 + 1/alpha) / (2 * alpha * np.log(1/rho)) # e.q. 3.4
        self.rho = rho
        self.u = u # claim 1 holds for probability at least 1 - e^{-u}
        self.v1 = v1
        self.v2 = v2
        self.C3 = C3
        self.C4 = C2 + 2 * np.log(n**2 * np.pi ** 2/6)
        self.D1 = D1

        super().__init__(arms_range, f, delta, k, n, reward_type, eta)

    def g(self,x):
        """In assumption A2"""
        # TODO: smoothness assumption, might needs to change later
        return x

    def V(self, h):
        """In claim 2"""
        # TODO
        temp = np.sqrt(2 * self.u + self.C4 + h * np.log(self.k) + 4 * self.D1 * np.log(1/self.g(self.v1 * self.rho ** h)))
        return 4 * self.g(self.v1 * self.rho ** h) * (temp + self.C3)

    def bvalue(self, x, t = None, reward = None):
        """
            Return bvalue of tree node x
        """
        if reward == None:
            reward = self.reward(x)
        self.samples[x].append(reward)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)

        self.update_posterior(x.data, reward)

        mu, var = self.m.predict(np.array([x.data]).reshape(1,1))
        if x.depth > 0:
            x_parent = x.parent
        else:
            x_parent = x
        mu_p, var_p = self.m.predict(np.array([x_parent.data]).reshape(1,1))

        term1 = mu[0,0] + self.beta_n * np.sqrt(var[0,0])
        term2 = mu_p[0,0] + self.beta_n * np.sqrt(var_p[0,0]) + self.V(x.depth - 1)
        U = np.min([term1, term2])

        return U + self.V(x.depth), self.beta_n * np.sqrt(var[0,0])

    def rec(self):
        # self.bvalues[self.root] = self.bvalue(self.root)

        ne = 0
        t = 1

        while ne <= self.n:
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    # self.bvalues[x] = self.bvalue(x)
                    self.bvalues[x] = np.inf

            selected_node = max(self.bvalues, key = self.bvalues.get)

            if selected_node not in self.betastd.keys():
                self.betastd[selected_node] = np.inf

            if self.betastd[selected_node] <= self.V(selected_node.depth) and selected_node.depth <= self.hmax:
                del self.bvalues[selected_node]
                if selected_node.depth > self.deepest_expanded_node.depth:
                    self.deepest_expanded_node = selected_node
                self.expand(selected_node)
            else:
                if selected_node in self.T_dict.keys():
                    self.T_dict[selected_node] += 1
                else:
                    self.T_dict[selected_node] = 1
                self.bvalues[selected_node], self.betastd[selected_node] = self.bvalue(selected_node)
                ne+=1

            t += 1

        return self.deepest_expanded_node


# ------------------------------------------------------------------------------
# Plot func 

    

def plot_tree(node, ax):
    ax.scatter(node.data, -node.depth, s=1.5, c = 'b')
    if len(node.children) > 0:
        for child in node.children:
            ax.plot([node.data, child.data], [-node.depth, - child.depth], c = 'gray', alpha = 0.5)
            plot_tree(child, ax)


def plot(arms_range, f, doo, axes):
    # fig, axes = plt.subplots(2, 1, figsize = (6,8), sharex=True)

    # data = []
    # neg_depth = []
    # for node in doo.evaluated_nodes:
    #     data.append(node.data)
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
run_StoOO = False
run_GPStoOO = True

n = 50 #150
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

    doo1 = DOO(arms_range, f, delta1, k, n, reward_type)

    rec_node1 = doo1.rec()
    print(rec_node1.data)
    print(rec_node1.depth)
    # print([node.data for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    doo2 = DOO(arms_range, f, delta1, k, n, 'ave')
    # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

    rec_node2 = doo2.rec()
    print(rec_node2.data)
    print(rec_node2.depth)
    # print(node.data for node in doo2.evaluated_nodes)
    # print(doo2.evaluated_fs)

    diff_node = []
    diff_f = []
    for i, node1 in enumerate(doo1.evaluated_nodes):
        for j, node2 in enumerate(doo2.evaluated_nodes):
            if node1.range[0] == node2.range[0] and node1.range[1] == node2.range[1]:
                # print('i: ', i)
                # print('j:', j)
                assert node1.data - node2.data < 1e-3
                diff_node.append(node1.data)
                diff_f.append(np.abs(doo1.evaluated_fs[i] - doo2.evaluated_fs[j]))
                # diff_dict[node1.data] = 
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
    sto1 = StoOO(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto1 = sto1.rec()
    print(rec_sto1.data)
    print(rec_sto1.depth)
    # print([node.data for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    sto2 = StoOO(arms_range, f, delta1, k, n, 'ave', eta)
    # doo2 = DOO(arms_range, f, delta2, k, n, reward_type)

    rec_sto2 = sto2.rec()
    print(rec_sto2.data)
    print(rec_sto2.depth)
    # print(node.data for node in doo2.evaluated_nodes)
    # print(doo2.evaluated_fs)
            
    print(len(sto1.evaluated_fs))
    print(len(sto2.evaluated_fs))


    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_two(arms_range, f, sto1, sto2, 'StoOO center v.s. ave')


if run_GPStoOO:
    sto1 = GPStoOO(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto1 = sto1.rec()
    print(rec_sto1.data)
    print(rec_sto1.depth)
    # print([node.data for node in doo1.evaluated_nodes])
    # print(doo1.evaluated_fs)
    # print()

    sto2 = StoOO(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto2 = sto2.rec()
    print(rec_sto2.data)
    print(rec_sto2.depth)

    sto3 = GPTree(arms_range, f, delta1, k, n, reward_type, eta)

    rec_sto3 = sto3.rec()
    print(rec_sto3.data)
    print(rec_sto3.depth)
            
    print(len(sto1.evaluated_fs))
    print(len(sto2.evaluated_fs))
    print(len(sto3.evaluated_fs))


    # plot(arms_range, f, doo1, 'doo1')
    # plot(arms_range, f, doo2, 'doo2')
    plot_three(arms_range, f, sto1, sto2, sto3, 'GPStoOO v.s. StoOO v.s. GPTree (center)')

        