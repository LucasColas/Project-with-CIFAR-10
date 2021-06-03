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
        print("yes, data : ", file)
        data = unpickle(path_file)
        print(len(data))
        print(data.keys())
        img = data[b'data'][i]
        img_r = img[0:1024].reshape(32,32)
        img_g = img[1024:2048].reshape(32,32)
        img_b = img[2048:].reshape(32,32)
        img_render = np.dstack((img_r, img_g, img_b))
        plt.imshow(img_render, interpolation='bicubic')
        plt.imshow(img_render)
        #plt.show()
        break

def render_several_images(file, num):
    for file in names:

        if "data" in file:
            for i in range(num):
                path_file = os.path.join(files_path, file)
                print("yes, data : ", file)
                data = unpickle(path_file)
                plt.subplot(330 + 1 + i)
                img = data[b'data'][i]
                img_r = img[0:1024].reshape(32,32)
                img_g = img[1024:2048].reshape(32,32)
                img_b = img[2048:].reshape(32,32)
                img_render = np.dstack((img_r, img_g, img_b))
                plt.imshow(img_render)
                print(i)
            break

    plt.show()
