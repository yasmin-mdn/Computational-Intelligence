# Q2_graded
# Do not change the above line.
# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from skimage.util import random_noise
from skimage.transform import resize

# Q2_graded
class hopfield:
    def __init__(self,train_files, in_shape): 
        self.Neuron_Num = in_shape[0] * in_shape[1]
        self.Weghits = np.zeros((self.Neuron_Num , self.Neuron_Num))

    def mat2vec(self,img):
      img = np.divide(img, 256)
      img_mean = np.mean(img)
      img = np.where(img < img_mean, -1, 1)
      img = img.flatten()
      return img

    def creat_w_matrix(self,train_files):
          #read image and convert it to Numpy array
          print("Importing images and creating weight matrix....")
          for img in train_files:
            img = self.mat2vec(img)
            for i in range(self.Neuron_Num):
              for j in range(i, self.Neuron_Num):
                if i != j:
                    w_ij = img[i] * img[j]
                    self.Weghits[i][j] += w_ij
                    self.Weghits[j][i] += w_ij                  
                else:
                    self.Weghits[i][j] = 0

          print("Weight matrix is done!!")


    def energy(self, S):
      energy=  -0.5 * np.matmul(np.matmul(S.T,self.Weghits), S)
      return energy



    def Predict(self, pattern, iterations, Async=False):
      n_img = self.mat2vec(pattern)

      if Async == False:
        energy = self.energy(n_img)
        for i in range(iterations):
          n_img = np.sign(np.matmul(self.Weghits, n_img))
          after_energy = self.energy(n_img)
          if energy == after_energy:
            return n_img
          energy = after_energy
        return n_img

# Q2_graded
(x_train, y_train), (_, _ )= fashion_mnist.load_data()
train_imgs=[]
top_idx = np.where( y_train == 0 )[0]
Trouser_idx= np.where( y_train == 1 )[0]
Pullover_idx= np.where( y_train == 2 )[0]
Dress_idx= np.where( y_train == 3 )[0]
Coat_idx= np.where( y_train == 4 )[0]
Sandal_idx= np.where( y_train == 5 )[0]
Shirt_idx= np.where( y_train == 6 )[0]
Sneaker_idx= np.where( y_train == 7 )[0]
Bag_idx= np.where( y_train == 8 )[0]
boot_idx= np.where( y_train == 9 )[0]
for i in [top_idx,Trouser_idx,Pullover_idx,Dress_idx,Coat_idx,Sandal_idx,Shirt_idx,Sneaker_idx,Bag_idx,boot_idx]:
  rand_idx = i[np.random.randint(0, len(i))]
  train_imgs.append(x_train[rand_idx])

# Q2_graded
test_imgs_10_percent=[]
test_imgs_30_percent=[]
test_imgs_60_percent=[]
for data in train_imgs:
    tmp= random_noise(data , mode='s&p',amount=0.1)
    test_imgs_10_percent.append(tmp)

for data in train_imgs:
     tmp= random_noise(data , mode='s&p',amount=0.3)
     test_imgs_30_percent.append(tmp)

for data in train_imgs:
     tmp= random_noise(data , mode='s&p',amount=0.6)
     test_imgs_60_percent.append(tmp)

# Q2_graded
net=hopfield(train_imgs,(28,28))
net.creat_w_matrix(train_imgs)
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( train_imgs[idx], (28, 28)),  cmap='gray')
    idx += 1

# Q2_graded
predicted_images_10 = []
predicted_images_30 = []
predicted_images_60= []
for img in test_imgs_10_percent:
  new_img = net.Predict(img, 2)
  predicted_images_10.append(new_img)

for img in test_imgs_30_percent:
  new_img = net.Predict(img, 2)
  predicted_images_30.append(new_img)


for img in test_imgs_60_percent:
  new_img = net.Predict(img, 2)
  predicted_images_60.append(new_img)


# Q2_graded
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_10[idx], (28, 28)),  cmap='gray')
    idx += 1
    
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_30[idx], (28, 28)),  cmap='gray')
    idx += 1

fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_60[idx], (28, 28)),  cmap='gray')
    idx += 1

print("acc for 10 percent noise in 28*28 net")
Acc(train_imgs,predicted_images_10,net)
print("acc for 30 percent noise in 28*28 net")
Acc(train_imgs,predicted_images_30,net)
print("acc for 60 percent noise in 28*28 net")
Acc(train_imgs,predicted_images_60,net)

# Q2_graded
train_imgs_32 = []
train_imgs_15 = []
for img in train_imgs:
  resized32_img = resize(img, (32, 32))
  resized15_img = resize(img, (15, 15))
  train_imgs_32.append(resized32_img)
  train_imgs_15.append(resized15_img)

test_imgs_10_percent_15=[]
test_imgs_30_percent_15=[]
test_imgs_60_percent_15=[]
for data in train_imgs_15:
    tmp= random_noise(data , mode='s&p',amount=0.1)
    test_imgs_10_percent_15.append(tmp)

for data in train_imgs_15:
     tmp= random_noise(data , mode='s&p',amount=0.3)
     test_imgs_30_percent_15.append(tmp)

for data in train_imgs_15:
     tmp= random_noise(data , mode='s&p',amount=0.6)
     test_imgs_60_percent_15.append(tmp)


test_imgs_10_percent_32=[]
test_imgs_30_percent_32=[]
test_imgs_60_percent_32=[]
for data in train_imgs_32:
    tmp= random_noise(data , mode='s&p',amount=0.1)
    test_imgs_10_percent_32.append(tmp)

for data in train_imgs_32:
     tmp= random_noise(data , mode='s&p',amount=0.3)
     test_imgs_30_percent_32.append(tmp)

for data in train_imgs_32:
     tmp= random_noise(data , mode='s&p',amount=0.6)
     test_imgs_60_percent_32.append(tmp)


# Q2_graded
net32=hopfield(train_imgs_32,(32,32))
net32.creat_w_matrix(train_imgs_32)
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( train_imgs_32[idx], (32, 32)),  cmap='gray')
    idx += 1


predicted_images_10_32 = []
predicted_images_30_32 = []
predicted_images_60_32= []
for img in test_imgs_10_percent_32:
  new_img = net32.Predict(img, 2)
  predicted_images_10_32.append(new_img)
  plt.figure()
  plt.imshow(np.reshape(new_img, (32, 32)))

for img in test_imgs_30_percent_32:
  new_img = net32.Predict(img, 2)
  predicted_images_30_32.append(new_img)


for img in test_imgs_60_percent_32:
  new_img = net32.Predict(img, 2)
  predicted_images_60_32.append(new_img)



fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_10_32[idx], (32, 32)),  cmap='gray')
    idx += 1
    
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_30_32[idx], (32, 32)),  cmap='gray')
    idx += 1

fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_60_32[idx], (32, 32)),  cmap='gray')
    idx += 1

print("acc for 10 percent noise in 32*32 net")
Acc(train_imgs_32,predicted_images_10_32,net32)
print("acc for 30 percent noise in 32*32 net")
Acc(train_imgs_32,predicted_images_30_32,net32)
print("acc for 60 percent noise in 32*32 net")
Acc(train_imgs_32,predicted_images_60_32,net32)


# Q2_graded
net15=hopfield(train_imgs_15,(15,15))
net15.creat_w_matrix(train_imgs_15)
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( train_imgs_15[idx], (15, 15)),  cmap='gray')
    idx += 1


predicted_images_10_15 = []
predicted_images_30_15 = []
predicted_images_60_15= []
for img in test_imgs_10_percent_15:
  new_img = net15.Predict(img, 2)
  predicted_images_10_15.append(new_img)
  plt.figure()
  plt.imshow(np.reshape(new_img, (15, 15)))

for img in test_imgs_30_percent_15:
  new_img = net15.Predict(img, 2)
  predicted_images_30_15.append(new_img)


for img in test_imgs_60_percent_15:
  new_img = net15.Predict(img, 2)
  predicted_images_60_15.append(new_img)



fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_10_15[idx], (15, 15)),  cmap='gray')
    idx += 1
    
fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_30_15[idx], (15, 15)),  cmap='gray')
    idx += 1

fig, axeslist = plt.subplots(ncols=5, nrows=2, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=0.5, bottom=0, top=0.2)
idx = 0
for i in range(2):
  for j in range(5):
    axeslist[i][j].imshow(np.reshape( predicted_images_60_15[idx], (15, 15)),  cmap='gray')
    idx += 1

print("acc for 10 percent noise in15*15 net")
Acc(train_imgs_15,predicted_images_10_15,net15)
print("acc for 30 percent noise in 15*15 net")
Acc(train_imgs_15,predicted_images_30_15,net15)
print("acc for 60 percent noise in 15*15 net")
Acc(train_imgs_15,predicted_images_60_15,net15)

