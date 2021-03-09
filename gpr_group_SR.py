import GPy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from gpr_group_model import GPRegression_Group
from generate_data import generate_data_func
from plot import plot_1d, plot_2d
from sklearn.cluster import KMeans
from gpr_group_test import generate_A, run_gprg
from generate_data import generate_data_func

# 2021/Mar/06

# Combine the GPR-Group prediction together with SR/SH algorithm.
# the groups are formulated using Kmeans algorithm (assume groups are fixed for now).
# Each round, one choose one group according to SR algorithm. 
# Goal: Recommend the best group.
# Baseline: groups are randomly chosen.

# We consider the same setting as shown in gpr_group_test.py

# Setting One:
# Fixed group in each round

x_shift = 0

# X_train_range_low = -3. + x_shift
# X_train_range_high = 3. + x_shift
# X_test_range_low = -3.5 + x_shift
# X_test_range_high = 3.5 + x_shift

X_train_range_low = -10. 
X_train_range_high = 10. 

num_train = 500
num_test = 2000
# num_train = X_train.shape[0]
# TODO: for now, assume num_train/num_group is integer
num_group = 50
num_element_in_group = int(num_train/num_group)
dim = 2

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
    def __init__(self, budget, num_arms, fixed_samples = None):
        self.budget = budget
        self.num_arms = num_arms
        self.fixed_samples = fixed_samples

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
            #print(self.fixed_samples[arm_idx])
            reward = self.fixed_samples[arm_idx][sample_idx]
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
                    if self.fixed_samples != None:
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
    def __init__(self, budget, num_arms, num_group, group_method = 'kmeans', fixed_samples = None):
        self.budget = budget
        self.num_arms = num_arms
        self.num_group = num_group
        self.fixed_samples = fixed_samples

        self.arms, self.f_train, self.Y_train = generate_data_func(self.num_arms ,self.num_arms ,dim=dim, X_train_range_low = X_train_range_low, X_train_range_high = X_train_range_high, x_shift = x_shift, func_type='sin')

        # choose RBF as default
        self.kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
        self.sample_groups = np.zeros((self.budget, self.num_arms))
        self.gpg = None
        self.mu = np.zeros((self.num_arms,))
        self.std = np.ones((self.num_arms,))
        self.rewards = []

    def sample(self):
        # sample group reward and add it into rewards record
        self.rewards.append(self.sample_groups[-1,:].dot(self.f_train) + np.random.randn()*0.05)

    def update(self):
        # update the posterior mean and std for each arm
        num_sample = len(self.rewards)

        if self.gpg is None:
            self.gpg = GPRegression_Group(self.arms,np.asarray(self.rewards).reshape(num_sample,1),
                                self.kernel, noise_var=0.005, A = self.sample_groups[:num_sample,:].reshape(num_sample, self.num_arms))
        else:
            self.gpg.set_XY_group(X=self.arms, Y= np.asarray(self.rewards).reshape(num_sample,1), A= self.sample_groups[:num_sample,:].reshape(num_sample, self.num_arms))

        self.mu, self.sigma = self.gpg.predict(self.arms)

    def form_group(self):
        # construct matrix A \in R^{g * n}
        # each row represents one group
        # the arms in the group are set to 1, otherwise 0
        kmeans = KMeans(n_clusters=self.num_group, init = 'k-means++', random_state= 0).fit(self.arms)
        group_idx = kmeans.labels_
        for idx,i in enumerate(group_idx):
            A[i, idx] = 1
        return A

    def max_ucb(self, t):
        # fill in the t^th sample group
        # the arms in the sample are set to 1, otherwise 0

        # method one: rec group with max ucb 

        # method two: rec top m arms with max ucb as a group 


        # self.sample_groups[t, :] = xxx

    def simulate(self):
        # REVIEW: for now we keep the group fixed 
        self.A = form_group()

        for t in range(budget):
            # all our rec and sample are in group level
            self.max_ucb(t)
            self.sample()
            self.update()

    def evaluation(self):
        # TODO: how to evaluate the pipeline?
        # 
            






