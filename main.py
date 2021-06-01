import pickle
import os
import matplotlib.pyplot as plt

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
        print(data[b'data'].shape)
        print(data[b'data'][0])
        plt.imshow(data[b'data'][5].reshape(32,32,3))
        plt.show()
        break
