
from matplotlib import pyplot as plt
import numpy as np

# Elements in this tuple below belongs to PIL.Image
from PIL import PngImagePlugin, BmpImagePlugin, GifImagePlugin, PpmImagePlugin, TiffImagePlugin, Image
PIL_CLASSES = (PngImagePlugin.PngImageFile, BmpImagePlugin.BmpImageFile, GifImagePlugin.GifImageFile,
               PpmImagePlugin.PpmImageFile, TiffImagePlugin.TiffImageFile, Image.Image)


# Show datas
def show_list_curv(y):
    """Y:np.ndarray(shape=(*,1))"""
    plt.clf()
    plt.figure()
    x = np.array(range(len(y))).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    plt.scatter(x, y)
    plt.show()
def show_2Dpoints(x1, x2):
    """X:np.ndarray(shape=(*,2) or shape=(2))"""
    plt.clf()
    plt.figure()
    plt.scatter(x1, x2)
    plt.show()
def show_3Dpoints(x1, x2, x3):
    """X1/X2/Y:np.ndarray(shape=(*,1))"""
    plt.clf()
    plt.figure()
    ax1 = plt.axes(projection = '3d')
    ax1.scatter3D(x1, x2, x3)
    plt.show()
# Draw the lines or surface
def show_2Dline(x, *Y):
    """X/Y:np.ndarray(shape=(*,1))"""
    plt.clf()
    plt.figure()
    if x==False:
        x = np.linspace(0,len(Y[0]),len(Y[0])).reshape(-1,1)
    for y in Y:
        plt.plot(x, y)
    plt.show()
def show_3Dsurface(x1, x2, *Y):
    """X1/X2:np.ndarray(shape=(*,1)); Y:np.ndarray(shape=(*,*))"""
    plt.clf()
    plt.figure()
    ax1 = plt.axes(projection = '3d')
    for i in Y:
        ax1.plot_surface(x1, x2, i, cmap='viridis')
    plt.show()



# Show images
try: import torch, torchvision
except ModuleNotFoundError: pass
def show_image(img):
    """
    show_image:
        function do half-automatically to show the picture by plt
        :param img: np.ndarray/PIL.Image.Image/torch.Tensor; all of this can be recognized
    demo:
        show_image(img) # img: ndarray/PIL_CLASSES/torch.tensor
    """
    def show_image_numpy(img: np.ndarray):
        assert len(img) == 3, "Wrong dimension of image, try 'print(img.shape)' to check the wrong."
        assert img.shape[-1] == 3, "Wrong depth of image, try 'print(img.shape)' to check the wrong."
        plt.clf()
        plt.imshow(img)
    def show_image_PIL(img: PIL_CLASSES):
        img = np.array(img)
        show_image_numpy(img)
    def show_image_tensor(img):
        img = torchvision.transforms.ToPILImage()(img)
        show_image_PIL(img)
    if isinstance(img, np.ndarray):
        show_image_numpy(img)
    elif isinstance(img, PIL_CLASSES):
        show_image_PIL(img)
    elif isinstance(img, torch.Tensor):
        show_image_tensor(img)
    else: # If no one above can work
        raise ValueError("The img input cannot be recognized.")

