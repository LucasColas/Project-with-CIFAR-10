import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers


files_path = r"dataset"
data_train_names = ["data_batch_1", "data_batch_2","data_batch_3", "data_batch_4"]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

names = os.listdir(files_path)
print(names)


def render_one_image(files_path, file, i):

        path_file = os.path.join(files_path, file)
        data = unpickle(path_file)

        img = data[b'data'][i]
        print(data[b'labels'][i])
        print(data[b'filenames'][i])

        img_r = img[0:1024].reshape(32,32)
        img_g = img[1024:2048].reshape(32,32)
        img_b = img[2048:].reshape(32,32)

        img_render = np.dstack((img_r, img_g, img_b))
        plt.imshow(img_render, interpolation='bicubic')

        plt.show()



def render_several_images(files_path, file, num):
    for i in range(num):

        path_file = os.path.join(files_path, file)
        data = unpickle(path_file)

        print(data[b'labels'][i])
        print(data[b'filenames'][i])

        plt.subplot(330 + 1 + i)

        img = data[b'data'][i]
        img_r = img[0:1024].reshape(32,32)
        img_g = img[1024:2048].reshape(32,32)
        img_b = img[2048:].reshape(32,32)

        img_render = np.dstack((img_r, img_g, img_b))
        plt.imshow(img_render)



    plt.show()


#render_several_images(files_path, names[2], 5)




def get_dataset(files_path, names):
    for name in names:
        X_train, Y_train = [], []
        X_valid, Y_valid = [], []
        X_test, Y_test = [], []

        if name in data_train_names:
            print("yes")

            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']

            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)
                X_train.append(data_preprocessed)
                Y_train.append(label)

        if name == "data_batch_5":
            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']
            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)
                X_valid.append(data_preprocessed)
                Y_valid.append(label)

        if name == "test_batch":
            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']
            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)
                X_test.append(data_preprocessed)
                Y_test.append(label)


        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        X_valid = np.asarray(X_valid)
        Y_valid = np.asarray(Y_valid)
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)

        return X_train, X_test, X_valid, Y_valid, X_test, Y_test

X_train, X_test, X_valid, Y_valid, X_test, Y_test = get_dataset(files_path, names)
