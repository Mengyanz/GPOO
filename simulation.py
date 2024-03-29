from typing import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
import GPy
from gpr_group_model import GPRegression_Group
from functools import partial
import pretty_errors

# np.random.seed(2021)

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
    opt_x: float or array
        optimal x 
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
    regret_list: list of regret for each time step

    T_dict: dict
        key: Tree instance 
        value: number of times the key node have been drawn
    samples: dict 
        key: Tree instance
        value: list of rewards observed at the key node
    rec_node: Tree instance
        indicator of the deepest expanded node, 
        the center of which will be recommended after the budget is run out.
    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma=0.0, opt_x = None) -> None:
        
        self.f = f # the function to be optimised, take arm feature as argument
        # if opt_x is None:
        #     self.opt_x = self.get_opt_x()
        # else:
        self.opt_x = opt_x 
        # print('opt x: ', self.opt_x)
        self.delta = delta # upper bound of diameters, a func of h
        
        self.n = n # sample budget
        self.k = k # k-ary tree, each node can have k children
        self.d = d # feature dim
        self.s = s
        self.reward_type = reward_type
        self.sigma = sigma

        assert reward_type in {'center', 'ave'}
        if reward_type == 'center':
            assert self.s == 1
        if reward_type == 'ave':
            assert self.s > 1
        self.A = np.ones(((1, self.s))) * (1.0/self.s)

        self.root = Tree()
        self.root.features = self.gene_feature(root_cell)
        self.root.center = self.center(root_cell)
        self.root.cell = root_cell 
        self.root.depth = 0 

        self.leaves = [self.root]

        self.evaluated_nodes = []
        self.evaluated_fs = []
        self.bvalues = {}
        self.regret_list = []

        self.T_dict = {} # key: node, value: number of times have been drawn
        self.samples = DefaultDict(list) # key: node, value: list of samples
        self.rec_node = self.root


    # def opt_x(self):
    #     size = 1000
        
    #     x = np.linspace(self.root.cell[0], self.root.cell[1], size)
        
    #     f_list = []
    #     for i in x:
    #         f_list.append(f(x))

    #     # REVIEW: we assume there is an unique opt point
    #     return np.argmax(f_list)

    def get_opt_x(self):
        """Empirical opt x.
        """
        size = 1000
        
        x = np.linspace(self.root.cell[0], self.root.cell[1], size).reshape(-1,1)
        f_list = self.f(x)
        
        # REVIEW: we assume there is an unique opt point
        return x[np.argmax(f_list)].reshape(-1,1)


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

        # FIXME: CHOOSE ONE VERSION!!
        # version 1: uniform split
        split_list = np.linspace(cell[0], cell[1], num = self.s + 1)
        for i in range(len(split_list)-1):
            features.append(self.center([split_list[i], split_list[i+1]]))

        # version 2: random sample
        # features = np.random.uniform(cell[0], cell[1], size = self.s)

        return np.asarray(features).reshape(self.s, self.d)


    def center(self, interval):
        """ 
            Return the center of a set of arms
        """
        return (interval[0] + interval[1])/2.0

    # def noiseles_reward(self,node):
    #     rewards = []
        
    #     for feature in node.features:
    #         rewards.append(self.f(feature))
    #     # REVIEW: need to change if use weighted sum
    #     return np.mean(rewards)

    def noiseles_reward(self,node):
        rewards = []
        
        for feature in node.features:
            feature = feature.reshape(1, self.d)
            rewards.append(self.f(feature))
        return np.mean(rewards)
    
    def reward(self, node):
        # REVIEW: to decide for non-gp case, whether to use bounded noise
        noise = np.random.normal(0, self.sigma)
        return self.noiseles_reward(node) + noise

    def sample(self,x):
        reward = self.reward(x)
        self.samples[x].append(reward)
        self.evaluated_nodes.append(x)
        self.evaluated_fs.append(reward)

        return reward

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
            self.T_dict[node] = 0

            self.leaves.append(node)
        self.leaves.remove(x)

    def regret(self):
        # print('opt x: ', self.opt_x)
        # print('f at opt x: ', self.f(self.opt_x))
        return self.f(self.opt_x)[0,0] - self.noiseles_reward(self.rec_node)

class Random(Base):
    def rec(self):
        self.k = self.n
        print(self.k)
        self.T_dict[self.root] = 0
        self.expand(self.root)
        reward = []
        for node in self.leaves:
            self.sample(node)
            self.T_dict[node] = 1
        self.rec_node = self.evaluated_nodes[np.argmax(self.evaluated_fs)]
        return self.regret()

class StoOO(Base):
    """
    Implementation of Stochastic Optimistic Optimisation algorithm 
    http://www.nowpublishers.com/articles/foundations-and-trends-in-machine-learning/MAL-038
    Fig 3.9

    Parameters
    ----------------------------------------------------------
    eta: float
        error probability, [0,1]
    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = None, eta=0.1) -> None:
        super().__init__(f, delta, root_cell, n, k, d, s, reward_type, sigma, opt_x)   
        self.eta = eta  

    def bvalue(self, x):
        """
            Return bvalue of tree node x
        """
        reward = self.sample(x)

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
            # only needs to update the bvalue of selected node
            self.bvalues[selected_node] = self.bvalue(selected_node)
            sample_count +=1
            
            thereshold = np.log(self.n**2/self.eta)/(2.0 * len(self.samples[selected_node]))
            # print('round ' +  str(i) + ' threshold ' + str(thereshold))
            if self.T_dict[selected_node] >= thereshold:
                del self.bvalues[selected_node]
                # FIXME: need to fix the case where there is more than one nodes in the deepest depth
                if selected_node.depth > self.rec_node.depth:
                    self.rec_node = selected_node
                self.expand(selected_node)

        return self.regret()

class GPOO(Base):
    """
    We extend StoOO to the case where f is sampled from GP. 

    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x = None, hmax = 4, **kwarg) -> None:
        super().__init__(f, delta, root_cell, n, k, d, s, reward_type, sigma, opt_x)

        self.X_list = []
        self.A_list = []
        self.Y_list = []
        self.sample_count = 0 
        self.hmax = hmax 

        # self.lengthscale = 0.1
        # self.kernel_var = 0.5
        # self.gp_noise_var = self.kernel_var * 0.05 # 1e-10 

        # self.lengthscale_bounds = [0.05, 10]
        # self.kernel_var_bounds = [0.05, 10]
        self.lengthscale_bounds = [0.01, 10]
        self.kernel_var_bounds = [0.05, 10]

        # self.kernel = GPy.kern.RBF(input_dim=1, 
        #                         variance=self.kernel_var, 
        #                         lengthscale=self.lengthscale)

        # self.f = self.gene_f() 
        # self.opt_x = self.get_opt_x()
            
        if kwarg.__len__() == 4:
            # for gp case, we need to input 4 paras in the following
            self.lengthscale = kwarg['lengthscale']
            self.kernel_var = kwarg['kernel_var']
            self.gp_noise_var = kwarg['gp_noise_var']
            self.opt_flag = kwarg['opt_flag']
        else: 
            print(kwarg.__len__())
            raise Exception

    def beta(self,t = 1):
        # return 0.5 * np.log(t)
        return 0.1 * np.log(np.pi**2 * t**2/(6 * 0.1))
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
        # print('x:', x.center)
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
        mu, var = self.m.predict(x.features, self.A)

        return np.sqrt(self.beta(self.sample_count)) * np.sqrt(var)

    def bvalue(self,x):
        # A = np.ones(((1, self.s))) * (1.0/self.s)
        mu, var = self.m.predict(x.features, self.A)
        return mu + np.sqrt(self.beta(self.sample_count)) * np.sqrt(var) + self.delta(x.depth)

    def update_bvalue(self):
        """update bvalue for all leaf nodes. 
        """
        for x in self.leaves:
            self.bvalues[x] = self.bvalue(x)

    def regression_eva(self):
        size = 100
        x = np.linspace(self.root.cell[0], self.root.cell[1], size).reshape(-1,1)

        f = self.f(x)
        mu, var = self.m.predict(x, A_ast = None)
        std = np.sqrt(var)

        node_centers = []
        for i, node in enumerate(self.evaluated_nodes):
            node_centers.append(node.center)

        plt.figure()
        plt.scatter(node_centers, self.evaluated_fs, label = 'obs')
        plt.plot(x, f, color = 'tab:orange', label = 'f')
        plt.plot(x, mu, color = 'tab:blue', label = 'pred')
        plt.fill_between(
            x.reshape(-1,), 
            (mu + self.beta() * std).reshape(-1,),
            (mu - self.beta() * std).reshape(-1,), 
            alpha = 0.3
            )
        plt.legend()
        # plt.ylim(-1,2)
        plt.savefig('reg' + str(self.sample_count) +'_'+self.reward_type+ '_opt' + str(self.opt_flag) + '.pdf')

    def rec(self):
        self.T_dict[self.root] = 0
        self.expand(self.root)
        for x in self.leaves:
            A,X,Y = self.add_obs(x)
            self.rec_node = x 
            regret = self.regret()
            self.regret_list.append(regret)


        kernel = GPy.kern.RBF(input_dim=1, 
                            variance=self.kernel_var, 
                            lengthscale=self.lengthscale) 
        self.m = GPRegression_Group(X, Y, kernel, A = A, noise_var=self.gp_noise_var)
        # self.m.optimize()
        self.update_bvalue()
        selected_node = max(self.bvalues, key = self.bvalues.get)

        while self.sample_count < self.n: # change n to sample budget
            # for x in self.leaves:
            #     if x not in self.evaluated_nodes:
            #         self.bvalues[x] = self.bvalue(x)
            #         # self.bvalues[x] = np.inf

            # print('# sample: ', self.sample_count)
            # print('leaves:')
            # for i in self.leaves:
            #     print(i.center)
            # print('bvalues:')
            # for key, value in self.bvalues.items():
            #     print(key.center)
            #     print(value)

            # print('selected node: ', selected_node.center)
            # print('################################')

            if self.delta(selected_node.depth) >= self.threshold(selected_node) and selected_node.depth <= self.hmax:
            # if self.T_dict[selected_node] >= self.threshold(selected_node):
                del self.bvalues[selected_node]
                if selected_node.depth > self.rec_node.depth:
                    self.rec_node = selected_node
                elif selected_node.depth == self.rec_node.depth:
                    if self.m.predict(selected_node.features, A_ast = self.A) > self.m.predict(self.rec_node.features, A_ast = self.A):
                        self.rec_node = selected_node

                self.expand(selected_node)

            selected_node = max(self.bvalues, key = self.bvalues.get)

            A,X,Y = self.add_obs(selected_node)

            kernel = GPy.kern.RBF(input_dim=1, 
                                variance=self.kernel_var, 
                                lengthscale=self.lengthscale) 
            kernel.lengthscale.constrain_bounded(self.lengthscale_bounds[0],self.lengthscale_bounds[1], warning=False)
            kernel.variance.constrain_bounded(self.kernel_var_bounds[0], self.kernel_var_bounds[1], warning=False)

            self.m = GPRegression_Group(X, Y, kernel, A = A, noise_var=self.gp_noise_var)
            # self.m.set_XY_group(X=X,Y=Y,A=A)
            if self.opt_flag:
                self.m.optimize()

            # print('*****************************')
            # print('kernel paras:')
            # print(self.m.kern.variance)
            # print(self.m.kern.lengthscale)
            # print(self.gp_noise_var)
            # print('*****************************')

            # if self.sample_count % 10 == 0:
            #     self.regression_eva()

            self.update_bvalue()
            
            # print('sample ', self.sample_count)
            # print('delta ', self.delta(selected_node.depth))
            # print('threshold ', self.threshold(selected_node))
            # if self.sample_count >=10:
            #     raise Exception
            regret = self.regret()
            self.regret_list.append(regret)

        # import pickle 
        # data_dict = {}
        # data_dict['X'] = X
        # data_dict['Y'] = Y
        # data_dict['A'] = A
        # with open('save_data.pickle', 'wb') as handle:
        #     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.regret_list

class GPTree(GPOO):
    """
        Algorithm in Shekhar et al. 2018
    """
    def __init__(self, f, delta, root_cell, n, k=2, d=1, s=1, reward_type = 'center', sigma = 0.1, opt_x= None, 
                alpha = 0.5, rho = 0.5, u = 2.0, v1 = 1.0, v2 = 1.0, C3 = 1.0, C2 = 1.0, D1=1, **kwarg) -> None:
        """
        alpha, rho (0,1)
        u >0
        0<v2<=1<=v1 
        C2,C3 > 0 (corollary 1)
        D1 >= 0 metric dimension (Defi 2)
        """
        hmax = np.log(n) * (1 + 1/alpha) / (2 * alpha * np.log(1/rho)) # e.q. 3.4
        super().__init__(f, delta, root_cell, n, k, d, s, reward_type, sigma,  opt_x, hmax, **kwarg)
        # TODO: might need to change constant rate
        self.beta_n = 0.25 * np.sqrt(np.log(n) + u)
        self.betastd = {}

        # Todo: the following parameters might need to be chosen more carefully
        
        self.rho = rho
        self.u = u # claim 1 holds for probability at least 1 - e^{-u}
        self.v1 = v1
        self.v2 = v2
        self.C3 = C3
        self.C4 = C2 + 2 * np.log(n**2 * np.pi ** 2/6)
        self.D1 = D1

    def g(self,x):
        """In assumption A2"""
        # TODO: smoothness assumption, might needs to change later
        return 0.1 * x

    def V(self, h):
        """In claim 2"""
        # TODO
        temp = np.sqrt(2 * self.u + self.C4 + h * np.log(self.k) + 4 * self.D1 * np.log(1/self.g(self.v1 * self.rho ** h)))
        return 4 * self.g(self.v1 * self.rho ** h) * (temp + self.C3)

    def threshold(self,x):
        mu, var = self.m.predict(x.features, self.A)
        return self.beta_n * np.sqrt(var) 

    def bvalue(self, x):
        mu, var = self.m.predict(x.features, self.A)
        term1 = mu + self.beta_n * np.sqrt(var)
        if x.depth > 0:
            mu_p, var_p = self.m.predict(x.parent.features, self.A)
            term2 = mu_p + self.beta_n * np.sqrt(var_p) + self.V(x.depth - 1)
            U = np.min([term1, term2])
        else:
            U = term1
        return U + self.V(x.depth)    

    def rec(self):
        # self.bvalues[self.root] = self.bvalue(self.root)
        

        while self.sample_count < self.n:
            print(self.sample_count)
            if self.sample_count == 0:
                
                A,X,Y = self.add_obs(self.root)

                kernel = GPy.kern.RBF(input_dim=1, 
                                    variance=self.kernel_var, 
                                    lengthscale=self.lengthscale) 
                self.m = GPRegression_Group(X, Y, kernel, A = A, noise_var=self.gp_noise_var)
                # self.m.optimize()
                self.update_bvalue()
                regret = self.regret()
                self.regret_list.append(regret)
            else:
                for x in self.leaves:
                    if x not in self.evaluated_nodes:
                        self.bvalues[x] = self.bvalue(x)
                        # self.bvalues[x] = np.inf

                selected_node = max(self.bvalues, key = self.bvalues.get)
                # print('selected node: ', selected_node)
        
                if self.threshold(selected_node) <= self.V(selected_node.depth) and selected_node.depth <= self.hmax:
                    # print('threshold: ', self.threshold(selected_node))
                    # print('self.v: ', self.V(selected_node.depth))
                    del self.bvalues[selected_node]
                    if selected_node.depth > self.rec_node.depth:
                        self.rec_node = selected_node
                    self.expand(selected_node)
                else:
                    # print('before add obs')
                    A,X,Y = self.add_obs(selected_node)
                    # print('finish add obs')
                    kernel = GPy.kern.RBF(input_dim=1, 
                                    variance=self.kernel_var, 
                                    lengthscale=self.lengthscale) 
                    kernel.lengthscale.constrain_bounded(self.lengthscale_bounds[0],self.lengthscale_bounds[1], warning=False)
                    kernel.variance.constrain_bounded(self.kernel_var_bounds[0], self.kernel_var_bounds[1], warning=False)

                    self.m = GPRegression_Group(X, Y, kernel, A = A, noise_var=self.gp_noise_var)
                    # self.m.set_XY_group(X=X,Y=Y,A=A)
                    if self.opt_flag:
                        self.m.optimize()
                    
                    self.update_bvalue()
                    regret = self.regret()
                    self.regret_list.append(regret)

        return self.regret_list
# ------------------------------------------------------------------------------
# Plot func 

def valid_plot_title(name):
    return name.replace('_', ' ')

def valid_plot_filename(name):
    return name.replace(' ', '_')

def plot_tree(node, T_dict, ax):
    if T_dict[node] == 0:
        c = 'b'
    elif T_dict[node] == 1:
        c = 'g'
    elif T_dict[node] > 1:
        c = 'r'
    ax.scatter(node.center, -node.depth, s=20, c = c)
    if len(node.children) > 0:
        for child in node.children:
            ax.plot([node.center, child.center], [-node.depth, - child.depth], c = 'gray', alpha = 0.5)
            plot_tree(child, T_dict, ax)

def plot(arms_range, f, oo, axes, name):
    # fig, axes = plt.subplots(2, 1, figsize = (6,8), sharex=True)

    # data = []
    # neg_depth = []
    # for node in oo.evaluated_nodes:
    #     data.append(node.features)
    #     neg_depth.append(- node.depth)
    # axes[0].scatter(data, neg_depth, s = 1)

    axes[0].set_title(name)
    plot_tree(oo.root, oo.T_dict, axes[0])
    
    size = 1000
    x = np.linspace(arms_range[0], arms_range[1], size)
    x = np.asarray(x).reshape(size, 1)
    axes[1].plot(x, f(x), c = 'r', alpha = 0.5, label = 'f')
    # plt.show()
    
    # if name == 'center':
    node_centers = []
    for i, node in enumerate(oo.evaluated_nodes):
        node_centers.append(node.center)
    axes[1].scatter(node_centers, oo.evaluated_fs, label = 'samples', color = 'grey', s = 5)
    # elif name == 'ave':
    #     for i, node in enumerate(oo.evaluated_nodes):
    #         axes[1].plot(node.cell, [oo.evaluated_fs[i], oo.evaluated_fs[i]], color = 'grey')
        
    if hasattr(oo, 'm'):
        mu, var = oo.m.predict(x, A_ast = None)
        std = np.sqrt(var)
        n = len(oo.evaluated_fs)
        beta = oo.beta(n)
        
        axes[1].plot(x, mu, color = 'tab:blue', label = 'predictions')
        axes[1].fill_between(
            x.reshape(-1,), 
            (mu + np.sqrt(beta) * std).reshape(-1,),
            (mu - np.sqrt(beta) * std).reshape(-1,), 
            alpha = 0.3
            )
    axes[1].legend()
    axes[1].set_ylim(-0.5,1.5)
    

def plot_two(arms_range, f, oo1, oo2, name, save_folder = ''):
    fig, axes = plt.subplots(2, 2, figsize = (12,8), sharex=True)
    plot(arms_range, f, oo1, axes[:, 0], 'S=1')
    plot(arms_range, f, oo2, axes[:, 1], 'S=10')
    # fig.suptitle(valid_plot_title(name))

    if hasattr(oo1, 'm'):
        assert oo1.opt_flag == oo2.opt_flag
        plt.savefig(save_folder + valid_plot_filename(name) + '_opt' + str(oo1.opt_flag) + '.pdf', bbox_inches='tight')
    else:
        plt.savefig(save_folder + valid_plot_filename(name) + '.pdf', bbox_inches='tight')

# def plot_three(arms_range, f, oo1, oo2, oo3, name = 'center'):
#     fig, axes = plt.subplots(2, 3, figsize = (12,8), sharex=True)
#     plot(arms_range, f, oo1, axes[:, 0])
#     plot(arms_range, f, oo2, axes[:, 1])
#     plot(arms_range, f, oo3, axes[:, 2])
#     fig.suptitle(name)
#     plt.savefig(name + '_oo.pdf')

def plot_regret(regret_dict, ax, n_repeat):

    color_dict = {}
    linestyle_dict = {}

    color_dict['GPOO S=1'] = 'tab:blue'
    color_dict['StoOO S=1'] = 'tab:orange'
    color_dict['GPTree S=1'] = 'tab:purple'
    color_dict['GPOO S=10'] = 'tab:brown'
    color_dict['StoOO S=10'] = 'tab:pink'

    color_dict['GPOO'] = 'tab:blue'
    color_dict['StoOO'] = 'tab:orange'
    color_dict['GPTree'] = 'tab:purple'
    # color_dict['Random'] = 'tab:green'

    # linestyle_dict['S=1'] = 'solid'
    # linestyle_dict['S=10'] = 'solid'

    # linestyle_str = [
    #  'solid', 'dotted', 'dashed', 'dashdot'
    #  ]  

    # if len(regret_dict)> len(linestyle_str):
    #     linestyle = 'solid'
    # i= 0

    for key, regret_list in regret_dict.items():
        
        # if len(regret_dict)<=len(linestyle_str):
        #     print(i)
        #     linestyle = linestyle_str[i]
        # if ' ' in key:
        #     alg, s = key.split(' ')
        # else: 
        #     alg = key
        #     s = 'S=1'
        regret_array = np.asarray(regret_list).reshape(n_repeat,-1)
        regret_mean = np.mean(regret_array, axis=0)
        regret_std = np.std(regret_array, axis=0)
        ax.plot(range(1,len(regret_mean)+1), regret_mean, label = str(key), color = color_dict[key]
        # , linestyle = linestyle_dict[s]
        )
        ax.fill_between(
            range(1,len(regret_mean)+1), 
            regret_mean + regret_std,
            regret_mean - regret_std,
            color = color_dict[key],
            alpha = 0.1)
        # i+=1
    ax.set_ylabel('Aggregated Regret')
    ax.set_xlabel('Budget')
    ax.set_ylim(-0.05, 1)
    ax.legend()

def plot_regret_one(regret_dict, name = 'Regret', budget = 50, n_repeat = 1, save_folder = '', plot_title = ''):
    fig, axes = plt.subplots(1, 1, figsize = (6,6))
    plot_regret(regret_dict, axes, n_repeat)
    if plot_title is not '':
        plt.title(valid_plot_title(plot_title))
    # budget = len(regret_list1[0])
    plt.savefig(save_folder + valid_plot_filename(name) + '.pdf', bbox_inches='tight')

# def plot_regret_two(regret_dict1, regret_dict2, name = 'Regret', budget = 50, n_repeat = 1):
#     fig, axes = plt.subplots(1, 2, figsize = (12,8), sharex=True)
#     plot_regret(regret_dict1, axes[0], n_repeat)
#     plot_regret(regret_dict2, axes[1], n_repeat)
#     fig.suptitle(valid_plot_title(name))
#     # budget = len(regret_list1[0])
#     plt.savefig(valid_plot_filename(name) + '.pdf', bbox_inches='tight')


            
