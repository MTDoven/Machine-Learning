
from sklearn.datasets import load_iris
from PIL import Image
import numpy as np
import random


def get_linear_data(size: tuple, seed: int = None) -> tuple:
    """
    get_linear_data:
        To Get Linear Data, it may be used to test machine learning algorithm.
        X is datas in a line with a little bias(+-0.25).
        y is target(=[1,2,3,...,n]) with no bias.
        :param size: (row, col); Only generate 2D data
        :param seed: int; To set the seed, so that get the same data
        :return: (X,y); X.shape=(row,col), y.shape=(row,1)
    demo:
        X, y = get_linear_data(size=(10, 2), seed=10)
        print(X); print(y)
    """
    assert len(size) == 2, "Only generate data in 2D"
    if seed is not None:
        random.seed(seed)
    row, col = size
    reshape_function = [lambda x: x * (random.random() - 0.5) * 2 for _ in range(col)]
    x_list = []
    for _ in range(col):
        x = np.array([i + (random.random() - 0.5) / 2 for i in range(row)]).reshape(-1, 1)
        x = x + random.randint(-row, row)
        x = random.choice(reshape_function)(x)
        x_list.append(x)
    X = np.concatenate(x_list, axis=1)
    y = np.array(range(row)).reshape(-1, 1)
    return X, y

def get_classify_data(size: tuple, classes: int, seed: int = None) -> tuple:
    """
    get_classify_data:
        To Get classify Data, it may be used to test machine learning algorithm.
        X is datas in a line with a little bias(+-0.167).
        y is target(=[0,0,0,...,k,k,k,...,n,n,n]), which just classify Step-shaped data
        :param size: (row, col); Only generate 2D data
        :param classes: int; the number of classes of target(y)
        :param seed: int; To set the seed, so that get the same data
        :return: (X,y); X.shape=(row,col), y.shape=(row,1)
    demo:
        X, y = get_classify_data(size=(10, 2), classes=2, seed=10)
        print(X); print(y)
    """
    X,y = get_linear_data(size=size, seed=seed)
    length = len(y)
    assert classes<=length, "The number of classes you want to divide is larger than the number of data"
    mirror = lambda x:(classes/length)*x
    for i in range(length):
        y[i] = int(mirror(y[i]))
    return X,y

def get_picture_data(mode="numpy", path="./Library/UtilsTools/DataExamples/PictureForKmeans.png"):
    """
    get_picture_data:
        Get a picture data
        :param mode: The type of return data, [numpy]/PIL/tensor
        :param path: The picture path, default PictureForKmeans.png
        :return: Up to your input
    demo:
        img = get_picture_data(mode="numpy", path="./demo.png")
        print(img)
    """
    try: # Use this not in the main path
        img = Image.open(path)
    except FileNotFoundError:
        path = "../"+path
        img = Image.open(path)
    if mode=="numpy":
        return np.array(img)
    elif mode=="PIL":
        img = np.array("img")
        img = Image.fromarray(img)
        return img
    elif mode=="tensor":
        import torchvision
        return torchvision.transforms.ToTensor()(img)
    else: # No one mode can fit the input
        raise ValueError("Please check the 'mode' you input.")

def get_iris_data():
    """
    get_iris_data:
        Data for SVM or classify
    demo:
        img = get_iris_data()
        print(img)
    """
    dataset, target = load_iris()['data'][:100, :2], load_iris()['target'][:100]
    target = target.reshape(-1,1)
    return dataset,target
