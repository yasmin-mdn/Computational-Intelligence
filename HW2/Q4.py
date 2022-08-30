# Q4_graded
# Do not change the above line.
from re import X
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

# Q4_graded
# Do not change the above line.
n=200
X_lst=[]
Y_lst=[]
def make_dataset(n):
  for i in range(0,n):
      x=random.uniform(0,1)
      X_lst.append(x)
      mu=random.uniform(-0.2,0.2)
      y=1/3 + 0.5*np.sin(3*x*np.pi)+mu
      Y_lst.append(y)

  X_train, X_test, y_train, y_test = train_test_split(X_lst, Y_lst,test_size=0.2)
  return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test= make_dataset(n)
X= np.array(X_train)
Y = np.array(y_train)
X_Test= np.array(X_test)
Y_Test= np.array(y_test)

# Q4_graded
model = Sequential()
model.add(Dense(128, input_shape=(1,), activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='Adam')
result=model.fit(X, Y, epochs=1000, batch_size=30,verbose=0)
y_predict = model.predict(X)
plt.plot(X, Y, 'bo', X, y_predict, 'go')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()
y_Test_predict = model.predict(X_Test)
plt.plot(X_Test, Y_Test, 'bo', X_Test, y_Test_predict, 'ro')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Test Value')
plt.show()

# Q4_graded
np.random.seed(1)
def rbf(x, center, radius):
  return np.exp(-1 / (2 * radius ** 2) * (x - center) ** 2)
def kmeans(X, k):
  clusters = np.random.choice(np.squeeze(X), size=k)
  prevClusters = clusters.copy()
  stds = np.zeros(k)
  converged = False

  while not converged:
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    for i in range(k):
      pointsForCluster = X[closestCluster == i]
      if len(pointsForCluster) > 0:
        clusters[i] = np.mean(pointsForCluster, axis=0)

    converged = np.linalg.norm(clusters - prevClusters) < 1e-6
    prevClusters = clusters.copy()

  distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
  closestCluster = np.argmin(distances, axis=1)

  clustersWithNoPoints = []
  for i in range(k):
    pointsForCluster = X[closestCluster == i]
    if len(pointsForCluster) < 2:
      clustersWithNoPoints.append(i)
      continue
    else:
      stds[i] = np.std(X[closestCluster == i])

  if len(clustersWithNoPoints) > 0:
    pointsToAverage = []
    for i in range(k):
      if i not in clustersWithNoPoints:
        pointsToAverage.append(X[closestCluster == i])

    pointsToAverage = np.concatenate(pointsToAverage).ravel()
    stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

  return clusters, stds

class RBF():
  def __init__(self, k):
    self.k = k
    self.w = np.random.randn(k)
    self.b = np.random.randn(1)

    self.centers = None
    self.stds = None

    self.iterations = None
    self.learning_rate = None

  def predict(self, x):
    y_predict = []
    for i in range(x.shape[0]):
      z = np.array([rbf(x[i], center, sigma) for center, sigma, in zip(self.centers, self.stds)])
      a = z.T.dot(self.w) + self.b
      y_predict.append(a)
    return np.array(y_predict)

  def update_parameters(self, a, error):
    self.w = self.w - self.learning_rate * a * error
    self.b = self.b - self.learning_rate * error

  def fit(self, x, y, iterations, learning_rate):
    self.iterations = iterations
    self.learning_rate = learning_rate
    self.centers, self.stds = kmeans(x, self.k)

    # SGD
    for i in range(self.iterations):
      for j in range(x.shape[0]):
        z = np.array([rbf(x[j], center, sigma) for center, sigma, in zip(self.centers, self.stds)])
        a = z.T.dot(self.w) + self.b
        difference = y[j] - a
        loss = (difference.flatten() ** 2) / 2
        error = (-1) * difference.flatten()
        self.update_parameters(z, error)

ITERATIONS = 500
LEARNING_RATE = 0.01

my_rbf = RBF(k=3)
my_rbf.fit(X, Y, ITERATIONS, LEARNING_RATE) 
y_predict = my_rbf.predict(X) 
plt.plot(X, Y, 'bo', X, y_predict, 'ro')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.show()

y_predict = my_rbf.predict(X_Test) 
plt.plot(X_Test, Y_Test, 'bo', X_Test, y_predict, 'go')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X_Test Value')
plt.show()

