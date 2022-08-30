# Q1_graded
# Do not change the above line.
import numpy as np
from copy import copy
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib as mpl
from collections import Counter
from operator import itemgetter
from scipy import spatial
from sklearn.metrics import pairwise_distances_argmin_min
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 500
np.random.seed(42)
import random

# Q1_graded
class SOM_2D():
    def __init__(self, init_lr, init_radius, radius_floor=0.6, weights_size=(20,20),  num_frames=10, lr_decay = 1, rad_decay=5, num_train=4500, num_test=500, image_dims="square", epochs=10, problem="mnist",):
        x = np.load(open("/content/mnist_x.npy", "rb"))
        y = np.load(open("/content/mnist_y.npy", "rb"))
        num_train_cols = x.shape[1]
        tot = np.concatenate((x, y), axis=1)
        np.random.shuffle(tot)
        x = tot[:,:num_train_cols]
        y = tot[:,num_train_cols:]
        sep1 = num_train 
        self.data_x = x[0:sep1]
        self.data_y = y[0:sep1]
        self.test_data_x = x[sep1:sep1+num_test]
        self.test_data_y = y[sep1:sep1+num_test]
        self.train_ind, self.test_ind = 0,0
        self.num_train = len(self.data_x)
        self.num_test = len(self.test_data_x)
        
        self.weight_dims = (weights_size[0], weights_size[1], self.data_x.shape[1])
        self.weights = self.init_weights(self.weight_dims)
        self.image_dims = (int(np.sqrt(self.data_x.shape[1])), int(np.sqrt(self.data_x.shape[1]))) if image_dims=="square" else image_dims
        self.it_limit = epochs*num_train
        self.num_frames = num_frames
        self.plot_interval = self.it_limit // self.num_frames
        self.init_lr = init_lr
        self.lr_func = lambda t: self.init_lr*np.exp(-lr_decay*t/self.it_limit)
        self.init_radius = init_radius
        self.radius_func = lambda t: max([self.init_radius*np.exp(-rad_decay*t/self.it_limit), radius_floor])
        self.neigh_func = lambda dist, rad: np.exp((-(dist**2)/(2*rad**2)))
        self.node_winners = {i*self.weight_dims[0]+j: [] for i in range(self.weight_dims[0]) for j in range(self.weight_dims[1])}
        self.top_nodes = {n:(0,0) for n in range(self.data_y.shape[1])}
        
    def get_next_train(self):
        x, y = self.data_x[self.train_ind], self.data_y[self.train_ind]
        self.train_ind += 1
        if self.train_ind >= self.num_train:
            self.train_ind = 0
        return x, y
    
    def get_next_test(self):
        x, y = self.test_data_x[self.test_ind], self.test_data_y[self.test_ind]
        self.test_ind += 1
        if self.test_ind >= self.num_test:
            self.test_ind = 0
        return x, y
    
    def init_weights(self, dims):
        weights = np.random.rand(*dims)
        return weights

    def get_ord_dist(self, x, y):
        # Calculate euclidean distance in topology space between two (x,y)-coordinates.
        return np.linalg.norm(np.array(x)-np.array(y))
        
    def get_dist(self, x, y):
        # Calculate euclidean distance between two vectors.
        return np.linalg.norm(x-y)
    
    def int_to_xy(self, i):
        # Convert an integer representation of weight index to (x,y)-coordinates.
        return np.unravel_index(i, (self.weight_dims[0], self.weight_dims[1]))
    
    def update_weights(self, input_vector, bmu_index, t):
        # Update the weights for the winning neuron and its neighborhood according to neighborhood function.
        for i in range(self.weight_dims[0]):
            for j in range(self.weight_dims[1]):
                lr = self.lr_func(t)
                radius = self.radius_func(t)
                dist = self.get_ord_dist(bmu_index, (i,j))
                lamb  =  self.neigh_func(dist, radius) 
                self.weights[i,j] = self.weights[i,j] + lr*lamb*(input_vector-self.weights[i,j])
    
    def get_winner_OLD(self, node, weights):
        return min([((i,j), self.get_dist(node, weights[i][j])) for i in range(self.weight_dims[0]) for j in range(self.weight_dims[1])], key=lambda x: x[1])

    def get_winner(self, node):
        ind, dist = pairwise_distances_argmin_min(node.reshape(1,-1), self.weights.reshape(self.weight_dims[0]*self.weight_dims[1],-1), metric="cosine")
        return self.int_to_xy(ind[0]), dist[0] #(ind[0]//self.weight_dims[0], ind[0]-(ind[0]//self.weight_dims[0])*self.weight_dims[1]))
    
    def get_win_class(self, node):
        top_labels = [x[0] for x in self.top_nodes.values()]
        ind, dist = pairwise_distances_argmin_min(node.reshape(1,-1), self.weights.reshape(self.weight_dims[0]*self.weight_dims[1],-1)[top_labels], metric="cosine")
        return (self.int_to_xy(top_labels[ind[0]])), dist[0]
        
    def one_hot_to_int(self, onehot):
        # Convert a one-hot-vector to integer.
        return np.where(onehot==1)[0][0]
    
    def get_most_common(self, l):
        # Get the most common value from a list.
        try:
            res = max((Counter(l).most_common(1)[0] for e in l), key=itemgetter(1))[0]
        except ValueError:
            res = "NaN"
        return res
    
    def run(self):
        # Initialize the plot that will be updated in animation.
        fig, axmat = plt.subplots(self.weight_dims[0], self.weight_dims[1], figsize=(20,20), squeeze=True)
        self.ims = []
        axs = axmat.flatten()
        for i in range(self.weight_dims[0]):
            for j in range(self.weight_dims[1]):
                axs[i*self.weight_dims[0]+j].set_yticklabels([])
                axs[i*self.weight_dims[0]+j].set_xticklabels([])
                im = axs[i*self.weight_dims[0]+j].imshow(self.weights[i,j].reshape(self.image_dims), cmap="gray", animated=True)
                self.ims.append(im)
        fig.suptitle("Iteration: 0 Lr:  Radius:  ", size=30)
        
        def update(frame, som):
            for iteration in range(som.plot_interval):
                # Select a random sample for training
                n, label = som.get_next_train()
                # Find the neuron closest to the sample.
                bmu_index, bmu_dist = som.get_winner(n)
                if frame==0:
                    break
                # Update weights for the winning neuron and its neighborhood
                som.update_weights(n, bmu_index, (som.plot_interval*frame)+iteration)
                # Update dict to keep track of which cases a neuron has won.
                som.node_winners[bmu_index[0]*som.weight_dims[0]+bmu_index[1]].append(som.one_hot_to_int(label))
            # Calculate the majority label for a neuron.
            som.node_labels = {k: som.get_most_common(v) for k, v in som.node_winners.items()}
            # Get the learning rate and radius for the current iteration.
            lr = som.lr_func(som.plot_interval*frame)
            radius = som.radius_func(som.plot_interval*frame)
            # Calculate train and test accuracy.
            train_acc = som.get_train_acc()
            test_acc = som.get_test_acc()
            # Update plot title
            fig.suptitle("Iteration: "+str(frame*som.plot_interval)+ " Lr: {:0.2f}".format(lr) +
                         " Radius: {:0.2f}".format(radius)+"\n Train acc: {:0.2f}".format(train_acc)+" Test acc: {:0.2f}".format(test_acc) , size=50)
            # Update images from weight arrays and set title according to majority label.
            for i in range(som.weight_dims[0]):
                for j in range(som.weight_dims[1]):
                    som.ims[i*som.weight_dims[0]+j].set_array(som.weights[i,j].reshape(som.image_dims))
                    axs[i*som.weight_dims[0]+j].set_title(str(som.node_labels[i*som.weight_dims[0]+j]), size=12, fontweight="bold")
            return som.ims
        
        animation = mpl.animation.FuncAnimation(fig, func=update, blit=True, interval=100, frames=self.num_frames+1, fargs=[self]);
        plt.close(fig)
        return animation
    
    def get_train_acc(self):
        # Calculation fraction correct predictions on the training data given the majority label as a classifier.
        num_correct = 0
        for iteration in range(self.num_train):
            n, label = self.get_next_train()
            bmu_index, bmu_dist = self.get_winner(n)
            if self.one_hot_to_int(label) == self.node_labels[bmu_index[0]*self.weight_dims[0]+bmu_index[1]]:
                num_correct += 1
        acc = num_correct/self.num_train
        return acc
    
    def get_test_acc(self):
        # Calculation fraction correct predictions on the test data given the majority label as a classifier.
        num_correct = 0
        for iteration in range(self.num_test):
            n, label = self.get_next_test()
            bmu_index, bmu_dist = self.get_winner(n)
            # Check if the input vectors label matches the winning neurons majority label.
            if self.one_hot_to_int(label) == self.node_labels[bmu_index[0]*self.weight_dims[0]+bmu_index[1]]:
                num_correct += 1
        acc = num_correct/self.num_test
        return acc
    
    def test(self):
        # Plot a random sample from the test set and display the input image and weight of the winning neuron for that input.
        test, label = self.get_test_sample()
        fig, (inp_ax, win_ax) = plt.subplots(2, figsize=(10,10))
        inp_ax.imshow(test.reshape((self.image_dims[0], self.image_dims[1])), cmap="gray")
        true_label = self.one_hot_to_int(label)
        inp_ax.set_title(str(true_label))
        winner, dist = self.get_winner(test)
        win_ax.imshow(self.weights[winner].reshape((self.image_dims[0], self.image_dims[1])), cmap="gray")
        winners_label = self.node_labels[winner[0]*self.weight_dims[0]+winner[1]]
        win_ax.set_title(str(winners_label))
        plt.show()


# Q1_graded
mnistparams = {}
mnistsom = SOM_2D(0.2, 2.5, 0.50,(20,20),10,1.1,5,4500,500,"square",10)
anim = mnistsom.run()
#HTML(anim.to_jshtml())

