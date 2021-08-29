from GPy.inference.latent_function_inference.posterior import PosteriorExact
import numpy as np
from GPy.util.linalg import pdinv, dpotrs, dpotri, symmetrify, jitchol, dtrtrs, tdot


class PosteriorExactGroup(PosteriorExact):
    def __init__(self, woodbury_chol=None, woodbury_vector=None, K=None, mean=None, cov=None, K_chol=None,
                 woodbury_inv=None, prior_mean=0, A = None):
        super(PosteriorExactGroup, self).__init__(woodbury_chol=woodbury_chol, woodbury_vector=woodbury_vector, 
                K=K, mean=mean, cov=cov, K_chol=K_chol, woodbury_inv=woodbury_inv, prior_mean=prior_mean)
        self.A = A
        

    def _raw_predict(self, kern, Xnew, A_ast, pred_var, full_cov=False):
        """
        pred_var: _predictive_variable, X_t (all X up to round t)
        """
        # print('PosteriorExactGroup _raw_predict')
        # NOTE: change Kx to AKx and add .dot(A_ast.T)
        # NOTE: 20210827 confirm mu and var (for self._woodbury_chol.ndim == 2 case)
        Kx = self.A.dot(kern.K(pred_var, Xnew)).dot(A_ast.T) # A_t k(X_t, X_\ast) A_\ast
        mu = np.dot(Kx.T, self.woodbury_vector) 
        # mu = A_\ast k(X_t, X_\ast)^T A_t^T (A_t k(X_t, X_t)A_t^T + sigma^2 I)^{-1} (Y_t - m)
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        if full_cov:
            Kxx = kern.K(Xnew) # k(X_ast, X_ast)
            # self._woodbury_chol Cholesky decomposition of A_t k(X_t, X_t)A_t^T + sigma^2 I
            if self._woodbury_chol.ndim == 2:
                # DTRTRS solves a triangular system of the form A * X = B  or  A**T * X = B, where A is a triangular matrix of order N, and B is an N-by-NRHS matrix.  A check is made to verify that A is nonsingular.
                tmp = dtrtrs(self._woodbury_chol, Kx)[0] # (A_t k(X_t, X_t)A_t^T + sigma^2 I)^{-1} k(X_ast, X_ast) -> v
                # tdot: returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles.
                var = A_ast.dot(Kxx - tdot(tmp.T)).dot(A_ast.T)
            elif self._woodbury_chol.ndim == 3:  # Missing data
                raise NotImplementedError('Need to be extended to group case!')
                var = np.empty((Kxx.shape[0], Kxx.shape[1], self._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = np.diag(A_ast.dot(kern.K(Xnew, Xnew)).dot(A_ast.T))
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                # tmp = tmp.dot(A_ast.T)
                # tmp = tmp.dot(A_ast)
                var = (Kxx - np.square(tmp).sum(0))[:, None]
            elif self._woodbury_chol.ndim == 3:  # Missing data
                raise NotImplementedError('Need to be extended to group case!')
                var = np.empty((Kxx.shape[0], self._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        return mu, var
