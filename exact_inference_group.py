from GPy.inference.latent_function_inference.exact_gaussian_inference import ExactGaussianInference
from posterior_group import PosteriorExactGroup
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np

log_2_pi = np.log(2*np.pi)

class ExactGaussianInferenceGroup(ExactGaussianInference):
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None, A = None):
        """
        TODO: to be changed
        Returns a Posterior class containing essential quantities of the posterior
        """
        print('ExactGaussianInferenceGroup inference:')
        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        # NOTE: change K to AKA^T
        if K is None:
            if A is None:
                A = np.identity(X.shape[0])
            K = A.dot(kern.K(X)).dot(A.T)
            

        Ky = K.copy()
        diag.add(Ky, variance+1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        # REVIEW: seems that log_marginal does not need to be changed
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        # REVIEW: since log_marginal does not change, the gradient does not need to change as well.
        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return PosteriorExactGroup(woodbury_chol=LW, woodbury_vector=alpha, K=K, A = A), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

