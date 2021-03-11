import numpy as np
import tensorflow as tf
import scipy.spatial as sp
import functools
import networks

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
                tf.transpose(self.sigma_star) @ tf.transpose(self.A(x)) \
                @ tf.linalg.solve(self.A(x) @ self.sigma @ tf.transpose(self.A(x)) + \
                self.noise_var*tf.eye(self.num_groups), self.A(x) @ self.sigma_star)

    @staticmethod
    def pairwise_dist_sq(A, B):
        """
        Computes pairwise squared distances between each elements of A and each 
            elements of B.
        Args:
            A (nparray(float)): (m,d) matrix
            B (nparray(float)): (n,d) matrix
        Returns:
        (m,n) nparray of floats representing matrix of pairwise distances
        """
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidean difference matrix
        D = tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0)
        return D

    def _init_kernel(self, kernel):
        """
        The kernel is currently the sum of the squared exponential, a trainable
        inner product of a neural network, and a squared exponential after 
        taking the L2 norms of the inputs. This will likely change in future.

        Args:
            kernel (fun, None): Use the kernel provided if not None.
        """
        if kernel is None:
            norm = lambda x: np.linalg.norm(x, axis=1).reshape((-1,1))
            kernel_rbf = lambda x1, x2: np.exp(\
                    -sp.distance.cdist(norm(x1), norm(x2))**2*SQ_EXP_PRECISION)\
                    .astype(np.float32)
            self.kernel_network = networks.MlpA(self.x.shape[1], 50, num_layers=1,
                    output_act = 'softmax')

            kernel_dist = lambda x1, x2: tf.exp(\
                    -self.pairwise_dist_sq(x1.astype(np.float32), 
                        x2.astype(np.float32))*SQ_EXP_PRECISION)
            kernel = lambda x1, x2: tf.transpose(self.kernel_network(x1)) @ \
                    self.kernel_network(x2) + kernel_dist(x1, x2) + \
                    kernel_rbf(x1, x2)

            self.kernel_params = self.kernel_network.get_weights()
            self.kernel = kernel
        else:
            self.kernel = kernel
            self.kernel_params = None

    def _init_optimiser(self, optimiser):
        """
        Initialise the tensorflow optimiser

        Args:
            optimiser(None, keras.optimizer): Use provided optimiser if not 
                None. Otherwise, default to Adam with SGD_LEARNING_RATE
        """
        if optimiser is None:
            optimiser = tf.keras.optimizers.Adam(SGD_LEARNING_RATE)
        self.optimiser = optimiser
   
    def _init_A(self, A):
        """
        Initialise the A function as a simple matrix by default.

        Args:
            A (None, keras.model): If None, A is a matrix. Otherwise,
                provide a keras model like an MLP.
        """
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
        """
        Calculate the required kernel matrices for GP regression.
        """
        self.sigma               = self.kernel(self.x, self.x)
        self.sigma_star          = self.kernel(self.x, self.x_star)
        self.sigma_star_star     = self.kernel(self.x_star, self.x_star)

    def train_step(self, x, return_A=False, return_ent=False):
        """
        Run one step of the optimiser on the posterior entropy objective.
        """
        with tf.GradientTape(persistent=False) as tape:
            self._calc_kernel_matrices()
            self.post_cov = lambda x: self.sigma_star_star - \
                    tf.transpose(self.sigma_star) @ tf.transpose(self.A(x)) \
                    @ tf.linalg.solve(self.A(x) @ self.sigma @ tf.transpose(self.A(x)) + \
                    self.noise_var*tf.eye(self.num_groups), self.A(x) @ self.sigma_star)
            post = self.post_cov(x)
            entropy = tf.linalg.logdet(post) + self.l2_lambda*\
                    sum([tf.norm(A_param)**2 for A_param in self.A_params])

        grads = tape.gradient(entropy, self.A_params+self.kernel_params)

        self.optimiser.apply_gradients(zip(grads, self.A_params+self.kernel_params))


        ret = []
        if return_A:
            ret = ret + \
                [[self.A_params[i].numpy() for i in range(len(self.A_params))]]
        if return_ent:
            ret = ret + [entropy]

        return ret


