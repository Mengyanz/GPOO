import numpy as np
import scipy.spatial as sp
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gp_ent_cluster
import networks

tf.config.run_functions_eagerly(False)

NOISE_VAR = 1.
NUM_GROUPS = 3
NUM_TEST = 12

## Make some data
c1 = np.random.normal(0, 0.2, (100, 2))
c2 = np.random.normal(-1, 0.2, (100, 2))
c3 = np.random.normal(1, 0.2, (100, 2))

X = np.vstack((c1, c2, c3))
x1 = np.linspace(-1, 1, 12)
x2 = np.linspace(-1, 1, 12)
X_star = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])

# Initialise the GP clusterer
A = networks.MlpA(X.shape[1], NUM_GROUPS)
cluster = gp_ent_cluster.GPEntCluster(NOISE_VAR, NUM_GROUPS, X, X_star, A=A)

# Minimise the entropy using gradient tape
c = 0
while True:
    _, entropy = cluster.train_step(X, True, True)

    np_A = cluster.A(X).numpy()
    print(entropy.numpy())
    if (c % 1000) == 0:
        plt.matshow(np_A.T @ np_A)
        plt.savefig('group_kernel.pdf')
        plt.close()
       
        np_A = np_A - np.amin(np_A)
        np_A = np_A / np.amax(np_A)
        plt.scatter(X[:, 0], X[:,1], c=np_A.T)
        plt.savefig('plt.pdf')
        plt.close()
    c = c + 1




