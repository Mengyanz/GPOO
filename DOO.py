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
    A: array
        group indicator matrix
    
    """
    def __init__(self) -> None:
        self.features = None
        self.center = None
        self.cell = None
        self.children = []
        self.depth = None
        self.parent = None
        self.A = None

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

        return np.asarray(features).reshape(self.s, self.d)

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
        self.leaves.remove(x)


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


class GPStoOO(StoOO):
    """
    We extend StoOO to the case where f is sampled from GP. 

    Parameter
    ---------------------------------------------
    kernel: GPy.kern instance. Default is RBF kernel. 
    # TODO: extend to support other kernels.

    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1, eta=0.1) -> None:
        super().__init__(f, delta, root_cell, n, k, d, s, reward_type, sigma, eta)

        # self.X = np.zeros((self.n * self.s, self.d))
        # self.A =  np.zeros((self.n, self.n * self.s))

        self.X_list = []
        self.A_list = []
        self.Y_list = []
        self.sample_count = 0 

    def sample(self,x):
        reward = self.reward(x)
        self.samples[x].append(reward)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)

        return reward

    def beta(self,t):
        # TODO: need to change (based on theo analysis)
        # return 0.5 * np.log(t)
        return 0.5 * np.log(np.pi**2 * t**2/(6 * self.eta))
        # return 1 

    def add_obs(self,x):
        """Sample reward of x and add observation x to X_list, A_list, Y_list 
        Return array A,X,Y which contains all previous observations
        """
        A_x = np.zeros((1, self.n * self.s))
        A_x[0, self.sample_count * self.s:(self.sample_count+1) * self.s] = 1.0/self.s * np.ones((1,self.s))
        self.A_list.append(A_x)
        self.X_list.append(x.features)
        reward = self.sample(x)
        if x in self.T_dict.keys():
            self.T_dict[x] += 1
        else:
            self.T_dict[x] = 1
        self.Y_list.append(reward)

        self.sample_count += 1
        A = np.asarray(self.A_list).reshape(self.sample_count, self.n * self.s)[:,:self.sample_count* self.s]
        X = np.asarray(self.X_list).reshape(self.sample_count * self.s, self.d)
        Y = np.asarray(self.Y_list).reshape(self.sample_count, 1)

        return A,X,Y

    def threshold(self,x):
        A = np.ones(((1, self.s))) * (1.0/self.s)
        mu, var = self.m.predict(x.features, A)

        return np.sqrt(self.beta(self.sample_count)) * np.sqrt(var)

    def bvalue(self,x):
        A = np.ones(((1, self.s))) * (1.0/self.s)
        mu, var = self.m.predict(x.features, A)
        return mu + np.sqrt(self.beta(self.sample_count)) * np.sqrt(var) + self.delta(x.depth)

    def update_bvalue(self):
        """update bvalue for all leaf nodes. 
        """
        for x in self.leaves:
            self.bvalues[x] = self.bvalue(x)

    def rec(self):
        self.expand(self.root)
        for x in self.leaves:
            A,X,Y = self.add_obs(x)

        print(A)
        print(X)
        print(Y)

        self.kernel = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=1.)
        self.m = GPRegression_Group(X, Y, self.kernel, A = A)
        # self.m.optimize()
        self.update_bvalue()

        while self.sample_count < self.n: # change n to sample budget
            for x in self.leaves:
                if x not in self.evaluated_nodes:
                    # self.bvalues[x] = self.bvalue(x)
                    self.bvalues[x] = np.inf

            selected_node = max(self.bvalues, key = self.bvalues.get)
            A,X,Y = self.add_obs(selected_node)
            self.m.set_XY_group(X=X,Y=Y,A=A)
            # self.m.optimize()
            self.update_bvalue()
            
            # print('round ' +  str(i) + ' threshold ' + str(thereshold))
            # if self.delta(selected_node.depth) >= self.threshold(selected_node):
            if self.T_dict[selected_node] >= self.threshold(selected_node):
                del self.bvalues[selected_node]
                # FIXME: need to fix the case where there is more than one nodes in the deepest depth
                if selected_node.depth > self.deepest_expanded_node.depth:
                    self.deepest_expanded_node = selected_node
                self.expand(selected_node)

        return self.deepest_expanded_node

class GPTree(GPStoOO):
    # FIXME: rewrite.
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

        self.update_posterior(x.features, reward)

        mu, var = self.m.predict(np.array([x.features]).reshape(1,1))
        if x.depth > 0:
            x_parent = x.parent
        else:
            x_parent = x
        mu_p, var_p = self.m.predict(np.array([x_parent.features]).reshape(1,1))

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

            
