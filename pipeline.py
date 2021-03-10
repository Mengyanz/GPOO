import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from gpr_group_model import GPRegression_Group
from generate_data import generate_data_func
from plot import plot_1d, plot_2d
from sklearn.cluster import KMeans
# from gpr_group_test import generate_A, run_gprg
from generate_data import generate_data_func

np.random.seed(1996)

# 2021/Mar/06

# Combine the GPR-Group prediction together with SR/SH algorithm.
# the groups are formulated using Kmeans algorithm (assume groups are fixed for now).
# Each round, one choose one group according to SR algorithm. 
# Goal: Recommend the best group.
# Baseline: groups are randomly chosen.

# 20210306: bandits setting does not really make sense to me
# why do we need to repeatedly select one group?
# maybe it makes more sense to have noise included for individual level

# 20210306: a naive SR does not make sense as wel
# since sampling a group multiple times in one round waste the samples
# given there are correlation between groups/individuals
# And the naive SR does not make use of any prediction information
# There might be literature gap in your mind: SR with correlated arm version
# We need a pure exploration group bandit algorithm (maybe for GP model precisely?)

class SR():
    def __init__(self, budget, num_arms, fixed_noise = None):
        self.budget = budget
        self.num_arms = num_arms
        self.fixed_noise = fixed_noise

        self.barlogK = 1.0/(1.0 + 1)
        for i in range(1, self.num_arms - 1.0 + 1):
            self.barlogK += 1.0/(self.num_arms + 1 - i)

        self.active_set = set(list(range(self.num_arms)))

    def cal_n_p(self,p):
        """Calculate n_p, the number of samples of each arm for phase p

        Parameters
        ----------------------------------------------------------------
        p: int
            current phase

        Return
        -----------------------------------------------------------------
        n_p: int
            the number of samples of each arm for phase p
        """
        n_p_float = 1.0/self.barlogK * (self.budget - self.num_arms)/ (self.num_arms + 1 - p)
        if n_p_float - int(n_p_float) > 0:
            n_p = int(n_p_float) + 1
        else:
            n_p = int(n_p_float)
        return n_p

    def sample(self, arm_idx, sample_idx = None):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        arm_idx: int
            the idx of arm with maximum ucb in the current round

        sample_idx: int
            sample from fixed sample list (for debug)
            if None: sample from env
            if int: sample as fixed_sample_list[sample_idx]
        
        Return
        ------------------------------
        reward: float
            sampled reward from idx arm
        """
        if sample_idx == None:
            reward = self.env[arm_idx].sample()
        else:
            #print('sample idx: ', sample_idx)
            #print(self.fixed_noise[arm_idx])
            reward = self.fixed_noise[arm_idx][sample_idx]
        self.sample_rewards[arm_idx].append(reward)
        self.left_budget -=1
        return reward

    def simulate(self):
        """Simulate experiments. 
        """
        n_last_phase = 0 # n_0
        # sample_count = 0
        for p in range(1, self.num_arms): # for p = 1, ..., K-1
            n_current_phase = self.cal_n_p(p)
            num_samples =  n_current_phase - n_last_phase

            # print('phase: ', p)
            # print('num_samples: ', num_samples)
            # print('active set: ', len(self.active_set))
            # sample_count += num_samples * len(self.active_set)
            # print('sample count: ', sample_count)

            # step 1
            for i in self.active_set:
                for j in range(num_samples):
                    if self.fixed_noise != None:
                        self.sample(i, len(self.sample_rewards[i]))
                    else:
                        self.sample(i)
            ss = {} # key: arm idx; value: empirical mean
            
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                
                ss[i] = np.mean(list(reward))

            self.active_set.remove(np.argsort(list(ss.values()))[0])
            # print(self.active_set)

            n_last_phase = n_current_phase

        self.rec_set = self.active_set
        # only works for 1.0 = 1
        assert len(self.rec_set) == 1.0

# It is more natural to use UCB type of algorithm, similar to GPUCB
# so we test UCB algorithm
# In the following we call the individual arms as "arm"
# the group arms as "group arm", we can only observe the sample reward from the selected group arm.

# Ways to form groups:
# 1. Super-arm (group) bandits:
#    form small groups (e.g. <=10 arms in each group) according to kmeans on feature space. 
#    Start from randomly (?, or maybe some reduce uncertainty method) 
#       picking one group and then from UCB on group level 
#    the group forming keep fixed and based on feature space 
# 2. Dynamic group based on UCB:
#    form large groups (e.g. >=100 arms in each group) according to kmeans on feature space.
#    Start from sampling once from each group as initialization 
#    Then selecting arms with top m (a small number e.g. m = 10) ucb as one group and observe the group label
#    the group forming change all the time and based on the prediction
# 3. follow similar idea as decision tree: form big groups (e.g. >= 100 arms in each group) at the init round, then in each group, ...

class UCB():
    def __init__(self, budget, num_arms, num_group, group_method = 'kmeans', fixed_noise = None):
        # TODO: implement fixed_noise
        
        self.budget = budget
        self.num_arms = num_arms
        self.num_group = num_group
        self.fixed_noise = fixed_noise

        self.arms, self.f_train, self.Y_train = generate_data_func(self.num_arms ,self.num_arms ,dim=dim, X_train_range_low = X_train_range_low, X_train_range_high = X_train_range_high, x_shift = x_shift, func_type='sin')

        # choose RBF as default
        self.kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
        self.sample_groups = np.zeros((self.budget, self.num_arms))
        self.gpg = None
        self.mu = np.zeros((self.num_arms,))
        self.sigma = np.ones((self.num_arms,))
        self.rewards = []

    def sample(self, t):
        # sample group reward and add it into rewards record
        sample = self.sample_groups[t,:].dot(self.f_train) + np.random.randn()* noise
        print(sample)

        self.rewards.append(sample)

    def update(self):
        # update the posterior mean and std for each arm
        num_sample = len(self.rewards)

        if self.gpg is None:
            self.gpg = GPRegression_Group(self.arms,np.asarray(self.rewards).reshape(num_sample,1),
            self.kernel, noise_var=0.005, A = self.sample_groups[:num_sample,:].reshape(num_sample, self.num_arms))
        else:
            self.gpg.set_XY_group(X=self.arms, Y= np.asarray(self.rewards).reshape(num_sample,1), A= self.sample_groups[:num_sample,:].reshape(num_sample, self.num_arms))

        # pred for indi
        self.mu, self.sigma = self.gpg.predict(self.arms)
        # pred for group
        self.group_mu, self.group_sigma = self.gpg.predict(self.arms, A_ast = self.A)

    def form_group(self):
        # construct matrix A \in R^{g * n}
        # each row represents one group
        # the arms in the group are set to 1, otherwise 0
        kmeans = KMeans(n_clusters=self.num_group, init = 'k-means++', random_state= 0).fit(self.arms)
        group_idx = kmeans.labels_
        A = np.zeros((self.num_group, self.num_arms))
        for idx,i in enumerate(group_idx):
            A[i, idx] = 1
        self.group_centers = kmeans.cluster_centers_
        return A

    def max_ucb(self, t, beta = 1):
        # fill in the t^th sample group
        # the arms in the sample are set to 1, otherwise 0

        # method one: rec group with max ucb 

        # method two: rec top m arms with max ucb as a group 
        if t == 0:
            # TODO: other methods than randomly choose a rec_idx initially 
            rec_idx = np.random.choice(list(range(self.num_group)))
        else:
            rec_idx = np.argmax(self.group_mu + beta * self.group_sigma)
        print(rec_idx)
        self.sample_groups[t,:] = self.A[rec_idx,:]
        # self.sample_groups[t, :] = xxx

    def simulate(self):
        # REVIEW: for now we keep the group fixed 
        self.A = self.form_group()

        for t in range(budget):
            # all our rec and sample are in group level
            self.max_ucb(t)
            self.sample(t)
            self.update()

    def evaluation(self):
        # TODO: how to evaluate the pipeline?

        # evaluate the prediction when budget is run out?
        print('Prediction for individual:')
        print('mean squared error: ', mean_squared_error(self.Y_train, self.mu))
        # print('Y train: ', self.Y_train)
        # print('mu: ', self.mu)
        print('r2 score: ', r2_score(self.Y_train, self.mu))

        print('Prediction for group (select A_ast = A):')
        group_train = self.A.dot(self.Y_train)
        print('mean squared error: ', mean_squared_error(group_train, self.group_mu))
        print('r2 score: ', r2_score(group_train, self.group_mu))

        
        plot_1d(self.arms, self.arms, self.f_train, self.Y_train, self.f_train, self.Y_train, self.mu, self.sigma, self.A, 
        self.group_centers, group_train, self.group_mu, self.group_sigma, 
        'gprg', self.num_group, grouping_method = 'kmeans')  
        
            
# Test
# We consider the same setting as shown in gpr_group_test.py

# Setting One:
# Fixed group in each round

x_shift = 0

# X_train_range_low = -3. + x_shift
# X_train_range_high = 3. + x_shift
# X_test_range_low = -3.5 + x_shift
# X_test_range_high = 3.5 + x_shift

X_train_range_low = -3. 
X_train_range_high = 3. 

budget = 5
num_arms = 30
# num_train = X_train.shape[0]
num_group = 5
dim = 1
noise = 1

# REVIEW: why we want a UCB type of algorithm? why does the uncertainty important? why we want to sample one arm multiple times?
# ? when the space is too large too cover (# num_arms > # num_budget)
# ? when the var of one arm is too large 

ucb = UCB(budget = budget, num_arms = num_arms, num_group = num_group, group_method = 'kmeans', fixed_noise = None)
ucb.simulate()
ucb.evaluation()

