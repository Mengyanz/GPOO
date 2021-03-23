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

# np.random.seed(1996)

# 2021/Mar/06

# Combine the GPR-Group prediction together with bandits algorithm.
# the groups are formulated using Kmeans algorithm (assume groups are fixed for now).
# Each round, one choose one group according to SR algorithm. 
# Goal: Recommend the best group.
# Baseline: groups are randomly chosen.

# 20210306: 
# why do we need to repeatedly select one group?
# maybe it makes more sense to have noise included for individual level

class Pipeline():
    """GPR-G + form group by cluster + bandits algorithm (UCB/SR)
    """
    def __init__(self, budget, num_arms, num_group, group_method = 'kmeans', noise = 0.1, fixed_noise = None, dynamic_grouping = False):
        # TODO: implement fixed_noise 
        self.budget = budget
        self.num_arms = num_arms
        self.num_group = num_group
        self.fixed_noise = fixed_noise
        self.group_method = group_method
        self.noise = noise
        self.dynamic_grouping = dynamic_grouping

        self.arms, self.f_train, self.Y_train = generate_data_func(self.num_arms ,self.num_arms ,dim=dim, X_train_range_low = X_train_range_low, X_train_range_high = X_train_range_high, x_shift = x_shift, func_type='sin')
        # self.idx_arms_dict = self.generate_idx_arms(self.arms)
        # print(self.idx_arms_dict)
        self.active_arm_idx = set(list(range(self.num_arms)))

        # choose RBF as default
        self.kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
        self.sample_groups = np.zeros((self.budget, self.num_arms))
        self.gpg = None
        self.mu = np.zeros((self.num_arms,))
        self.sigma = np.ones((self.num_arms,))
        self.rewards = []

    # def generate_idx_arms(self, arms):
    #     idx_arms_dict = {}
    #     for idx, arm in enumerate(arms):
    #         # print(arm)
    #         arm = str(arm)
    #         arm = list(arm)
    #         print(arm)
    #         idx_arms_dict[idx] = arm
    #         idx_arms_dict[arm] = idx
    #     return idx_arms_dict

    def sample(self, t):
        # sample group reward and add it into rewards record
        sample = self.sample_groups[t,:].dot(self.f_train) + np.random.randn()* self.noise
        # print(sample)

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

    def form_group(self, num_group):
        # construct matrix A \in R^{g * n}
        # each row represents one group
        # the arms in the group are set to 1, otherwise 0

        # print(self.active_arm_idx)
        sorted_active_arm_idx = np.asarray(np.sort(list(self.active_arm_idx)))
        data = self.arms[sorted_active_arm_idx,:]
        if self.group_method == 'kmeans':
            kmeans = KMeans(n_clusters=num_group, init = 'k-means++', random_state= 0).fit(data)
            group_idx = kmeans.labels_
            A = np.zeros((num_group, self.num_arms))   
            for i, idx in enumerate(sorted_active_arm_idx):
                A[group_idx[i], idx] = 1

            # check whether we need to change group centers code
            self.group_centers = kmeans.cluster_centers_

            if self.dynamic_grouping:
                # print('chaning group idx.')
                self.active_group_idx = set(list(range(num_group)))
            return A
        elif self.group_method == 'identity':
            self.group_centers = data
            return np.eye(N = data.shape[0])

    def evaluation(self):
        # TODO: how to evaluate the pipeline?

        # evaluate the prediction when budget is run out?
        print('Prediction for individual:')
        print('mean squared error: ', mean_squared_error(self.Y_train, self.mu))
        # print('Y train: ', self.Y_train)
        # print('mu: ', self.mu)
        print('r2 score: ', r2_score(self.Y_train, self.mu))
        
        if not self.dynamic_grouping:
            print('Prediction for group (select A_ast = A):')
            group_train = self.A.dot(self.Y_train)
            print('mean squared error: ', mean_squared_error(group_train, self.group_mu))
            print('r2 score: ', r2_score(group_train, self.group_mu))

        
        # plot_1d(self.arms, self.arms, self.f_train, self.Y_train, self.f_train, self.Y_train, self.mu, self.sigma, self.A, 
        # self.group_centers, group_train, self.group_mu, self.group_sigma, 
        # 'gprg', self.num_group, grouping_method = 'kmeans')  
        
# 20210315：
# We implement SR in pipeline:
# 1. reject group in each round, instead of reject arms
# 2. instead of using sample mean, use posterior mean
# 3. (we might want to cook up posterior std as a guideline for designing sample size)
# 4. another (open) questions is: how to deal with arm set which are changing (I imagine in our setting, later on we would like to dynamic form the groups)? Now in SR, the sample size is pre-allocated so only works for fixed set of arms. 

class SR_Fixed_Group(Pipeline):
    def __init__(self, budget, num_arms, num_group, group_method = 'kmeans', noise = 0.1, fixed_noise = None, dynamic_grouping = False):
        super().__init__(budget, num_arms, num_group, group_method, noise, fixed_noise, dynamic_grouping)
        self.barlogK = 1.0/(1.0 + 1)
        # REVIEW: for now, we form the "arm" of pre-defined fixed groups
        for i in range(1, self.num_group):
            self.barlogK += 1.0/(self.num_group + 1 - i)
        self.active_group_idx = set(list(range(self.num_group))) # for active groups
        # print('init:', self.active_group_idx)

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
        n_p_float = 1.0/self.barlogK * (self.budget - self.num_group)/ (self.num_group + 1 - p)
        if n_p_float - int(n_p_float) > 0:
            n_p = int(n_p_float) + 1
        else:
            n_p = int(n_p_float)
        return n_p

    def simulate(self):
        """Simulate experiments. 
        """
        self.A = self.form_group(self.num_group)

        n_last_phase = 0 # n_0
        # sample_count = 0
        t = 0
        for p in range(1, self.num_group): # for p = 1, ..., K-1
            n_current_phase = self.cal_n_p(p)
            num_samples =  n_current_phase - n_last_phase

            # print('phase: ', p)
            # print('num_samples: ', num_samples)
            # print('active set: ', len(self.active_group_idx))
            # sample_count += num_samples * len(self.active_group_idx)
            # print('sample count: ', sample_count)
            # print('num samples: ', num_samples)
            # step 1
            for i in self.active_group_idx:
                for j in range(num_samples):
                    # REVIEW: keep the same structure in pipeline 
                    self.sample_groups[t,:] = self.A[i,:]
                    self.sample(t)
                    t += 1
            self.update()
            # print('active set: ', self.active_group_idx)
            
            # print('group mu: ', self.group_mu)
            # TODO: remove the smallest one in active set
            group_mu = self.group_mu.copy()
            while True:
                min_idx = np.argmin(group_mu.reshape(len(group_mu),))
                if min_idx in self.active_group_idx:
                    # remove the rejected group 
                    self.active_group_idx.remove(min_idx)
                    # remove all arms in the reject group
                    for arm_idx, indicator in enumerate(self.A[min_idx,:]):
                        if indicator == 1:
                            self.active_arm_idx.remove(arm_idx)
                    break
                else:
                    group_mu[min_idx] = 1e5 # set to a very large value

            if self.dynamic_grouping:
                # TODO: does not work for now
                self.A = self.form_group(self.num_group - p)

            # sorted_group_mu_idx = np.argsort(self.group_mu.reshape(self.num_group,))
            # print('sorted group mu idx: ', sorted_group_mu_idx)
            # for i, idx in enumerate(np.sort(sorted_group_mu_idx)[::-1]):
            #     if idx in self.active_group_idx:
            #         # remove the group in active set with smallest pred mean
            #         print('remove idx: ', i)
            #         self.active_group_idx.remove(i)
            #         break
                
            

            n_last_phase = n_current_phase

        self.rec_set = self.active_group_idx
        # print(self.rec_set)
        print(np.asarray(self.f_train)[np.asarray(self.A[np.asarray(list(self.rec_set))][0], dtype=bool)])
        # only works for 1.0 = 1
        assert len(self.rec_set) == 1.0

        rec_idx = np.argmax(self.mu)
        print('rec arm index: ', rec_idx, ' with mean: ', self.f_train[rec_idx])

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

class UCB_Fixed_Group(Pipeline):
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
        # print(rec_idx)
        self.sample_groups[t,:] = self.A[rec_idx,:]
        # self.sample_groups[t, :] = xxx

    def simulate(self):
        # REVIEW: for now we keep the group fixed 
        self.A = self.form_group(self.num_group)

        for t in range(budget):
            # all our rec and sample are in group level
            self.max_ucb(t)
            self.sample(t)
            self.update()     

        rec_idx = np.argmax(self.mu)
        print('rec arm index: ', rec_idx, ' with mean: ', self.f_train[rec_idx])
       
# Test
# We consider the same setting as shown in gpr_group_test.py

# Setting One:
# Fixed group in each round

x_shift = 0

# X_train_range_low = -3. + x_shift
# X_train_range_high = 3. + x_shift
# X_test_range_low = -3.5 + x_shift
# X_test_range_high = 3.5 + x_shift

X_train_range_low = -5. 
X_train_range_high = 5. 

budget = 100
num_arms = 200
# num_train = X_train.shape[0]
num_group = 50
dim = 2
group_noise = 0.1 # now group label only has group noise
indi_noise = 0.1
run_UCB_Fixed_Group = False
run_UCB_non_Group = True
run_SR_Fixed_Group = True
run_SR_Dynamic_Group = True
run_SR_non_Group = False

# REVIEW: why we want a UCB type of algorithm? why does the uncertainty important? why we want to sample one arm multiple times?
# ? when the space is too large too cover (# num_arms > # num_budget)
# ? when the var of one arm is too large 

if run_UCB_Fixed_Group:
    print('GP-UCB Fixed Group:')
    ucb_fg = UCB_Fixed_Group(budget = budget, num_arms = num_arms, num_group = num_group, group_method = 'kmeans', noise = group_noise, fixed_noise = None)
    ucb_fg.simulate()
    ucb_fg.evaluation()

if run_UCB_non_Group:
    print('GP-UCB non group:')
    ucb_ng = UCB_Fixed_Group(budget = budget, num_arms = num_arms, num_group = num_arms, group_method = 'identity', noise = indi_noise, fixed_noise = None)
    ucb_ng.simulate()
    ucb_ng.evaluation()

if run_SR_Fixed_Group:
    print('GP SR Fixed Group:')
    sr_fg = SR_Fixed_Group(budget = budget, num_arms = num_arms, num_group = num_group, group_method = 'kmeans', noise = group_noise, fixed_noise = None)
    sr_fg.simulate()
    sr_fg.evaluation()

if run_SR_Dynamic_Group:
    print('GP SR Dynamic Group:')
    sr_fg = SR_Fixed_Group(
        budget = budget, num_arms = num_arms, num_group = num_group, group_method = 'kmeans', 
        noise = group_noise, fixed_noise = None, dynamic_grouping = True
        )
    sr_fg.simulate()
    sr_fg.evaluation()

if run_SR_non_Group:
    print('GP SR non Group:')
    sr_ng = SR_Fixed_Group(budget = budget, num_arms = num_arms, num_group = num_arms, group_method = 'identity', noise = indi_noise, fixed_noise = None)
    sr_ng.simulate()
    sr_ng.evaluation()

