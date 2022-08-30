# Q2_graded
# Do not change the above line.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
np.random.seed(0)
# Dataset
X,Y = datasets.make_circles(n_samples=576, shuffle=True, noise=0.25, random_state=None, factor=0.4)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='plasma')
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('Data')
plt.show()

# Q2_graded
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

# Q2_graded
X=X.T
Y=np.reshape(Y,(1,576))

# Q2_graded
m = 400
i_count = 1
def for_prop(X, parameters, n_layers,le_r):
    cache = {}
    outp = X

    for ind in range(n_layers - 1):
        b_name = "b" + str(ind)
        w_name = "w" + str(ind)
        a_name = "a" + str(ind)
        z_name = "z" + str(ind)
        cache[z_name] = np.dot(parameters[w_name], outp) + parameters[b_name]
        if (ind != n_layers - 2):
            cache[a_name] = np.tanh(cache[z_name])
        else:
            cache[a_name] = sigmoid(cache[z_name])
            tempp = cache[a_name]
            tempp[tempp>0.5]=1
            tempp[tempp<=0.5]=0
            accuracy = np.sum(Y[0] == tempp ) / len(Y[0])
            global i_count
            if i_count % 100 == 0:
              print("accuracy of %i_count: %f" % (i_count, accuracy))
            i_count+=1
        outp = cache[a_name]
    return outp, cache


def ReLu(inp):
    return np.maximum(inp, 0)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def back_prop(parameters, cache, X, Y, n_layers,le_r):
    m = X.shape[1]
    dzlast = []

    for ind in range(n_layers - 2, -1, -1):
        alast = "a" + str(ind - 1)

        b_name = "b" + str(ind)
        w_name = "w" + str(ind)
        a_name = "a" + str(ind)
        z_name = "z" + str(ind)

        output = cache[a_name]
        if (ind == 0):
            output = X.T
        else:
            output = cache[alast].T

        curw = parameters[w_name]
        curb = parameters[b_name]
        if (ind == n_layers - 2):
            dz = cache[a_name] - Y
            dw = 1.0 / m * np.dot(dz, output)
            db = 1.0 / m * np.sum(dz, axis=1, keepdims=True)
            dzlast = dz
            parameters[w_name] = curw - le_r * dw
            parameters[b_name] = curb - le_r * db
        else:
            wlast = "w" + str(ind + 1)
            w_cur = parameters[wlast]
            w_cur = np.einsum('ij->ji', w_cur)
            dz = np.dot(w_cur, dzlast) * (1 - np.power(cache[a_name], 2))
            dw = 1.0 / m * np.dot(dz, output)
            db = 1.0 / m * np.sum(dz, axis=1, keepdims=True)
            dzlast = dz
            parameters[w_name] = curw - le_r * dw
            parameters[b_name] = curb - le_r * db

    return parameters


def mlp(X, Y, inp,le_r=0.01, num_it=10000):
    # np.random.seed(3)
    inp.append(1)
    inp.insert(0, 2)
    weights = []
    bias = []
    parameters = {}
    n_layers = len(inp)
    for ind in range(len(inp) - 1):
        bias.append(np.zeros((inp[ind + 1], 1)))
        b_name = "b" + str(ind)
        w_name = "w" + str(ind)
        parameters[b_name] = bias[ind]
        parameters[w_name] = np.random.randn(inp[ind + 1], inp[ind])
    for i in range(0, num_it):
        output, cache = for_prop(X, parameters, n_layers,le_r)
        parameters = back_prop(parameters, cache, X, Y, n_layers,le_r)
    return parameters


def predict(parameters, X, n,le_r):
    output, cache = for_prop(X, parameters, n,le_r)
    predictions = (output > 0.5)

    return predictions


inputs = [3,3]
parameters = mlp(X, Y, inputs,0.1, num_it=10000)

plot_decision_boundary(lambda x: predict(parameters, x.T, len(inputs),le_r=0.01), X, Y.flatten())
plt.title("Decision Boundary ")

