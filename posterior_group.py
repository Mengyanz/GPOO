from GPy.inference.latent_function_inference.posterior import PosteriorExact
import numpy as np
from GPy.util.linalg import pdinv, dpotrs, dpotri, symmetrify, jitchol, dtrtrs, tdot


class PosteriorExactGroup(PosteriorExact):
    def __init__(self, woodbury_chol=None, woodbury_vector=None, K=None, mean=None, cov=None, K_chol=None,
                 woodbury_inv=None, prior_mean=0, A = None):
        super(PosteriorExactGroup, self).__init__(woodbury_chol=woodbury_chol, woodbury_vector=woodbury_vector, 
                K=K, mean=mean, cov=cov, K_chol=K_chol, woodbury_inv=woodbury_inv, prior_mean=prior_mean)
        self.A = A
        

    def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):
        print('PosteriorExactGroup _raw_predict')
        # NOTE: change Kx to AKx
        Kx = self.A.dot(kern.K(pred_var, Xnew))
        mu = np.dot(Kx.T, self.woodbury_vector)
        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)
        if full_cov:
            Kxx = kern.K(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = Kxx - tdot(tmp.T)
            elif self._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0], Kxx.shape[1], self._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = (Kxx - np.square(tmp).sum(0))[:, None]
            elif self._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0], self._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        return mu, var
