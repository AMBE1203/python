from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
np.random.seed(2)

X = np.random.rand(1000,1)
y = 4 + 3*X + .2*np.random.rand(1000, 1) # thêm nhiễu

# building xBar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T,y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ', w_lr.T)

# display result
w = w_lr
w0 = w[0][0]
w1 = w[1][0]
x0 = np.linspace(0,1,2, endpoint=True)
y0 = w0 + w1*x0

# draw the fitting line
plt.plot(X.T,y.T,'b.')
plt.plot(x0,y0,'y',linewidth = 2)
plt.axis([0,1,0,10])
plt.show()

def grad(w):
    N = Xbar.shape[0]
    return 1/ N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w),2)**2

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] += eps
        g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
    return g

def check_grad(w,cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print('Checking gradient...', check_grad(np.random.rand(2,1), cost, grad))
