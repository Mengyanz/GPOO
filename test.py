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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
print(list(chunks(range(10, 75), 10)))

for i in list(chunks(range(10, 75), 10)):
    print(i[-1])

print(np.asarray([[0,1], [1,2], [2,3]]).shape)

print(np.linspace(0, 9, num = 10 + 1))

print(np.random.normal(0,0))


