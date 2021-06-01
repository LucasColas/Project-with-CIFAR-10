import pickle
import os

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
        print(data[b'labels'])
        break
