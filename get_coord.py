import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

# 2048x2048.jpg  size: 2048 x 2048

def on_press(event):
    print("my position:" ,event.button,event.xdata, event.ydata)

if __name__ == '__main__':
    fig = plt.figure()
    img = Image.open('2048x2048.jpg')
    plt.plot([9, 1024, 2037, 1024], [1024, 9, 1024, 2037], 'r*')
    plt.imshow(img, animated= True)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()
