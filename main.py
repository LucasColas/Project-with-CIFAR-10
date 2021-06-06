import pickle
import os
import matplotlib.pyplot as plt
import numpy as np



files_path = r"dataset"

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

def order_images(dataset):
    data = dataset[b'data']
    labels = dataset[b'labels']
    labeled_data = []
    for data, label in zip(data, labels):
        data_preprocessed = np.reshape(32,32,3)/255
        labeled_data.append(data_preprocessed)

    labeled_data = np.asarray(labeled_data)
    return labeled_data


for name in names:
    if "data" in name:
        path_file = os.path.join(files_path, name)
        dataset = unpickle(path_file)
        print(len(dataset[b'data']))
        order_images(dataset)
        break
