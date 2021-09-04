import GPy
from gpr_group_model import GPRegression_Group
import numpy as np
import pretty_errors
import matplotlib.pyplot as plt
# GPy.plotting.change_plotting_library('plotly')

'''

np.random.seed(2021)
sample_size = 5
size = 3

# lengthscale = 0.2 # 1
# variance = 0.5 # 0.5
# noise_var = variance * 0.05  # 1e-10

lengthscale = 0.1
variance = 0.5
noise_var = 1e-10


# X = np.random.uniform(0, 1., (sample_size, 1))
X = np.array([0.05, 0.2, 0.4, 0.65, 0.9]).reshape(sample_size,1)
# Y = np.sin(X) + np.random.randn(sample_size, 1)*0.05
Y = np.array([0.9, 0.1, 0.95, 0.05, 0.98]).reshape(sample_size,1)

kernel = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
model = GPy.models.GPRegression(X,Y,kernel, noise_var=noise_var)

testX = np.linspace(0, 1, 100).reshape(-1, 1)
posteriorTestY = model.posterior_samples_f(testX, size=size)
simY, simMse = model.predict(testX)

# for i in range(size):
#     plt.plot(testX, posteriorTestY[:,:,i], label = 'sample posterior')
# plt.plot(X, Y, 'ok', markersize=5)
# plt.plot(testX, simY, label = 'posterior mean')
# plt.plot(testX, simY - 3 * simMse ** 0.5, '--g')
# plt.plot(testX, simY + 3 * simMse ** 0.5, '--g')
# plt.legend()
# plt.savefig('posterior_f.png')
'''
import pickle
with open('save_data.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)

# model.set_XY(data_dict['x'], data_dict['y'])
model = GPRegression_Group(data_dict['X'], data_dict['Y'],kernel, A=data_dict['A'], noise_var=noise_var)
model.optimize()
pred, var = model.predict(testX,A_ast=None)
std = np.sqrt(var)
print(std)
plt.scatter(data_dict['X'], data_dict['Y'])
plt.plot(testX, simY, label = 'posterior mean')
plt.plot(testX, pred, label = 'pred')
plt.fill_between(
            testX.reshape(-1,), 
            np.asarray(pred + std).reshape(-1,),
            np.asarray(pred - std).reshape(-1,), 
            alpha = 0.3
            )
plt.legend()
plt.show()



