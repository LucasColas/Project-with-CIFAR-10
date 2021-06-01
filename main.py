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
        print(data[b'labels'][600])
        print(data[b'filenames'][600])
        plt.figure(figsize = (0.5,0.5))
        plt.imshow(data[b'data'][600].reshape(32,32,3), aspect='auto')
        plt.show()
        break
