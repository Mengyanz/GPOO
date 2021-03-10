import numpy as np
import tensorflow as tf
import scipy.spatial as sp
import functools

SQ_EXP_PRECISION    = 8
SGD_LEARNING_RATE   = 0.01

class GPEntCluster(object):
    def __init__(self, noise_var, num_groups, x, x_star, kernel=None, 
            optimiser=None, A=None, l2_lambda=0):
        """
        Args:
            noise_var (float): variance of iid Gaussian noise on group 
                observations.
            num_groups (int): the number of rows of the matrix A.
            x (nparray(float)): the data that requires clustering.
            x_star (nparray(float)): the test points over which to minimise
                the entropy of the gp posterior predictive.
            kernel (None, function): the kernel function. Defaults to
                squared exponential if None is provided.
            optimiser (None, keras optimiser): the optimiser that minimises
                the entropy of the posterior predictive gp. If None, default
                to Adam with learning rate SGD_LEARNING_RATE
            A (model): A model with trainable parameters that defines the
                matrix A. Defaults to a simple matrix with num_groups rows.
            l2_lambda (float): parameter for l2 regularisation, i.e. weight
                decay.
        """
        # Keep some instance variables
        self.noise_var  = noise_var
        self.num_groups = num_groups
        self.x          = x
        self.x_star     = x_star
        self.l2_lambda  = l2_lambda

        # Initialise kernel, optimiser and A
        self._init_kernel(kernel)
        self._init_optimiser(optimiser)
        self._init_A(A)

        # Initialise the posterior covariance
        self._calc_kernel_matrices()
        self.post_cov = lambda x: self.sigma_star_star - \
                self.sigma_star.T @ tf.transpose(self.A(x)) \
                @ tf.linalg.solve(self.A(x) @ self.sigma @ tf.transpose(self.A(x)) + \
                self.noise_var*tf.eye(self.num_groups), self.A(x) @ self.sigma_star)

    def _init_kernel(self, kernel):
        if kernel is None:
            kernel = lambda x1, x2: np.exp(\
                    -sp.distance.cdist(x1, x2)**2*SQ_EXP_PRECISION)\
                    .astype(np.float32)
        self.kernel = kernel

    def _init_optimiser(self, optimiser):
        if optimiser is None:
            optimiser = tf.keras.optimizers.Adam(SGD_LEARNING_RATE)
        self.optimiser = optimiser
   
    def _init_A(self, A):
        if A is None:
            A_mat = np.random.normal(0, 1, 
                    (self.num_groups, self.x.shape[0])).astype(np.float32)
            A_mat = tf.Variable(A_mat, name='A', trainable=True)
            self.A = lambda x: A_mat
            self.A_params = [A_mat]
        else:
            self.A = A
            self.A_params = self.A.get_weights()

    def _calc_kernel_matrices(self):
        self.sigma               = self.kernel(self.x, self.x)
        self.sigma_star          = self.kernel(self.x, self.x_star)
        self.sigma_star_star     = self.kernel(self.x_star, self.x_star)

    def train_step(self, x, return_A=False, return_ent=False):
        with tf.GradientTape(persistent=False) as tape:
            post = self.post_cov(x)
            entropy = tf.linalg.logdet(post) + self.l2_lambda*\
                    sum([tf.norm(A_param)**2 for A_param in self.A_params])

        grads = tape.gradient(entropy, self.A_params)
        self.optimiser.apply_gradients(zip(grads, self.A_params))
        
        ret = []
        if return_A:
            ret = ret + \
                [[self.A_params[i].numpy() for i in range(len(self.A_params))]]
        if return_ent:
            ret = ret + [entropy]

        return ret


