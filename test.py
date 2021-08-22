import GPy
import numpy as np
import pretty_errors
# GPy.plotting.change_plotting_library('plotly')

num = 1

X = np.array([1]).reshape(num,1)
Y = np.sin(X) + np.random.randn(num,1)*0.05
# print(Y)
num = 10
X2 = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(num,1)
Y2 = np.sin(X) + np.random.randn(num,1)*0.05

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)
# m.optimize()
print(m.predict(X))

m.set_XY(X2,Y2)
# m.optimize(messages = False)
print(m.predict(X))




