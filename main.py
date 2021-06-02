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

for file in names:
    if "data" in file:
        path_file = os.path.join(files_path, file)
        print("yes, data : ", file)
        data = unpickle(path_file)
        print(len(data))
        print(data.keys())
        img = data[b'data'][900]
        img_r = img[0:1024].reshape(32,32)/255.0
        img_g = img[1024:2048].reshape(32,32)/255.0
        img_b = img[2048:].reshape(32,32)/255.0
        img_render = np.dstack((img_r, img_g, img_b))
        plt.imshow(img_render, interpolation='bicubic')
        plt.show()
        break
