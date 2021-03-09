import numpy as np
import scipy.spatial as sp
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#tf.enable_eager_execution()
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

# Set up the kernel
sq_exp = lambda x1, x2: np.exp(-sp.distance.cdist(x1, x2)**2*8).astype(np.float32)
sigma               = sq_exp(X, X)
sigma_star          = sq_exp(X, X_star)
sigma_star_star     = sq_exp(X_star, X_star)

# Set up tf minimizer
optimizer = tf.keras.optimizers.SGD(0.1)

# Set up the A matrix as a tensorflow variable to optimise
A = np.random.normal(0, 1, (NUM_GROUPS, X.shape[0])).astype(np.float32)
A = tf.Variable(A, name='A', trainable=True)

# Set up the entropy, the log of the determinant of the covariance matrix 
# of the posterior
post_cov = lambda x: sigma_star_star - sigma_star.T @ tf.transpose(A) @ tf.linalg.solve(\
            A @ sigma @ tf.transpose(A) + NOISE_VAR*tf.eye(NUM_GROUPS),
            A @ sigma_star)

# Minimise the entropy using gradient tape
c = 0
while True:
    with tf.GradientTape(persistent=False) as tape:
        post = post_cov(None)
        entropy = tf.linalg.logdet(post) + 0.8*tf.norm(tf.transpose(A)@A)

    grads = tape.gradient(entropy, [A])
    optimizer.apply_gradients(zip(grads, [A]))

    np_A = A.numpy()
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




