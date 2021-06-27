import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications.vgg16 import VGG16

VGG_model = VGG16(weights="imagenet", include_top=False)
VGG_model.summary()

files_path = r"dataset"
data_train_names = ["data_batch_1", "data_batch_2","data_batch_3", "data_batch_4"]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

names = os.listdir(files_path)



def render_one_image(files_path, file, i):

        path_file = os.path.join(files_path, file)
        data = unpickle(path_file)

        img = data[b'data'][i]
        #print(data[b'labels'][i])
        #print(data[b'filenames'][i])

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

        #print(data[b'labels'][i])
        #print(data[b'filenames'][i])

        plt.subplot(330 + 1 + i)

        img = data[b'data'][i]
        img_r = img[0:1024].reshape(32,32)
        img_g = img[1024:2048].reshape(32,32)
        img_b = img[2048:].reshape(32,32)

        img_render = np.dstack((img_r, img_g, img_b))
        plt.imshow(img_render)



    plt.show()


#render_several_images(files_path, names[2], 9)




def get_dataset(files_path, names):

    X_train, Y_train = [], []
    X_valid, Y_valid = [], []
    X_test, Y_test = [], []

    for name in names:


        if name in data_train_names:


            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']

            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)

                X_train.append(data_preprocessed)

                label_one_hot = [0 for i in range(10)]
                label_one_hot[label-1] = 1
                Y_train.append(label_one_hot)

        if name == "data_batch_5":
            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']
            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)
                X_valid.append(data_preprocessed)

                label_one_hot = [0 for i in range(10)]
                label_one_hot[label-1] = 1
                Y_valid.append(label_one_hot)

        if name == "test_batch":
            path_file = os.path.join(files_path, name)
            dataset = unpickle(path_file)
            data = dataset[b'data']
            labels = dataset[b'labels']
            for data, label in zip(data, labels):
                data_preprocessed = data.reshape(32,32,3)
                label_one_hot = [0 for i in range(10)]
                label_one_hot[label-1] = 1
                X_test.append(data_preprocessed)
                Y_test.append(label_one_hot)


    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_valid = np.asarray(X_valid)
    Y_valid = np.asarray(Y_valid)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = get_dataset(files_path, names)

for i in range(100):
    #print("X train : ", X_train[i], "Label train : ", Y_train[i], "X valid : ", X_valid[i], "Y valid :", Y_valid[i])
    pass

model = models.Sequential()


model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3,3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3,3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

print(X_train.shape, X_valid.shape,X_test.shape )
model.compile(optimizer=optimizers.SGD(lr=0.06),loss="categorical_crossentropy", metrics=["acc"])
history = model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_valid, Y_valid))
model.save("training.h5")


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs, val_acc,'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
