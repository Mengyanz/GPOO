from GPy.inference.latent_function_inference.exact_gaussian_inference import ExactGaussianInference
from posterior_group import PosteriorExactGroup
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np

log_2_pi = np.log(2*np.pi)

class ExactGaussianInferenceGroup(ExactGaussianInference):
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None, A = None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        The comments below corresponds to Alg 2.1 in GPML textbook.
        """
        # print('ExactGaussianInferenceGroup inference:')
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
            K = A.dot(kern.K(X)).dot(A.T) # A_t k(X_t, X_t) A_t^T
        else:
            raise NotImplementedError('Need to be extended to group case!')
            

        Ky = K.copy()
        diag.add(Ky, variance+1e-8) # A_t k(X_t, X_t)A_t^T + sigma^2 I

        # pdinv: 
        # Wi: inverse of Ky
        # LW: the Cholesky decomposition of Ky -> L
        # LWi: the Cholesky decomposition of Kyi (not used)
        # W_logdet: the log of the determinat of Ky
        Wi, LW, LWi, W_logdet = pdinv(Ky) 

        # LAPACK: DPOTRS solves a system of linear equations A*X = B with a symmetric
        # positive definite matrix A using the Cholesky factorization
        # A = U**T*U or A = L*L**T computed by DPOTRF.
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        # so this gives 
        # (A_t k(X_t, X_t)A_t^T + sigma^2 I)^{-1} (Y_t - m)

        # Note: 20210827 confirm the log marginal likelihood 
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        # REVIEW: since log_marginal does not change, the gradient does not need to change as well.
        # FIXME: confirm the gradient update is correct
        # dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)
        dL_dK = 0.5 * A.T.dot((tdot(alpha) - Y.shape[1] * Wi)).dot(A)
        # print('dL_dK shape', dL_dK.shape)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return PosteriorExactGroup(woodbury_chol=LW, woodbury_vector=alpha, K=K, A = A), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

