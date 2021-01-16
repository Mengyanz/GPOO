# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from GPy.core import GP
from GPy import likelihoods
from GPy import kern
from exact_inference_group import ExactGaussianInferenceGroup

class GPRegression_Group(GP):
    """
    Gaussian Process model for regression with group labels only

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed group values
    :param kernel: a GPy kernel, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)
    :param A: g * n group indicator matrix
        if it is Non, then it is initialized as identity matrix (n * n),
        then it is the same as GPR.

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None, A = None):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Gaussian(variance=noise_var)

        super(GPRegression_Group, self).__init__(X, Y, kernel, likelihood, name='GP regression group', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)
        self.inference_method = ExactGaussianInferenceGroup()
        self.A = A

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        return GPRegression(gp.X, gp.Y, gp.kern, gp.Y_metadata, gp.normalizer, gp.likelihood.variance.values, gp.mean_function)

    def to_dict(self, save_data=True):
        model_dict = super(GPRegression_Group,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPRegression_Group"
        return model_dict

    @staticmethod
    def _from_dict(input_dict, data=None):
        import GPy
        input_dict["class"] = "GPy.core.GP"
        m = GPy.core.GP.from_dict(input_dict, data)
        return GPRegression_Group.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

    def parameters_changed(self):
        """
        TODO: to be changed
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        print('GPRG trigger parameters changed.')
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata, A = self.A)
        # REVIEW: whether we need to change likelihood (Gaussian), kern, mean_function class?
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        # FIXME: gradient need change!
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`, this is the objective function of the model being optimised
        """
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        TODO: to be changed
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)
        return mu, var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None,
                likelihood=None, include_likelihood=True):
        """
        TODO: to be changed
        Predict the function(s) at the new point(s) Xnew. This includes the
        likelihood variance added to the predicted underlying function
        (usually referred to as f).

        In order to predict without adding in the likelihood give
        `include_likelihood=False`, or refer to self.predict_noiseless().

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the
                           likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :param include_likelihood: Whether or not to add likelihood noise to
                                   the predicted underlying latent function f.
        :type include_likelihood: bool

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False,
                 Nnew x Nnew otherwise

            If full_cov and self.input_dim > 1, the return shape of var is
            Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return
            shape is Nnew x Nnew. This is to allow for different normalizations
            of the output dimensions.

        Note: If you want the predictive quantiles (e.g. 95% confidence
        interval) use :py:func:"~GPy.core.gp.GP.predict_quantiles".
        """

        # Predict the latent function values
        mean, var = self._raw_predict(Xnew, full_cov=full_cov, kern=kern)

        if include_likelihood:
            # now push through likelihood
            if likelihood is None:
                likelihood = self.likelihood
            mean, var = likelihood.predictive_values(mean, var, full_cov,
                                                     Y_metadata=Y_metadata)

        if self.normalizer is not None:
            mean = self.normalizer.inverse_mean(mean)

            # We need to create 3d array for the full covariance matrix with
            # multiple outputs.
            if full_cov & (mean.shape[1] > 1):
                var = self.normalizer.inverse_covariance(var)
            else:
                var = self.normalizer.inverse_variance(var)

        return mean, var

