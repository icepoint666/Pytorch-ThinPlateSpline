import numpy as np
import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

import ThinPlateSpline as TPS

# 2048x2048.jpg  size: 2048 x 2048

def on_press(event):
    p = np.array([
        [693.55, 531.26],
        [1069.85, 1243.04],
        [1243.74, 1238.69],
        [472.82, 664.85],
        [552.50, 1460.07],
        [1021.03, 368.02],
        [1260.78, 1571.90],
        [93.16, 911.26],
        [234.85, 914.14],
        [383.34, 1140.97],
        [375.46, 853.36],
        [256.73, 597.61],
        [338.32, 502.28],
        [754.67, 337.95],
        [1120.42, 1797.99],
        [1521.97, 1655.66],
        [1371.15, 1832.87],
        [1522.78, 1315.94],
        [1116.38, 754.82],
        [1165.72, 1162.44],
        [1024.00, 1024.00]])

    v = np.array([
        [121.52, 25.00],
        [142.31, -10.74],
        [150.81, -10.63],
        [109.60, 18.24],
        [113.58, -22.72],
        [139.92, 34.87],
        [153.25, -28.63],
        [45.29, -25.83],
        [95.26, 5.30],
        [105.86, -6.01],
        [104.90, 8.46],
        [96.95, 16.70],
        [96.81, 27.64],
        [122.71, 37.11],
        [147.14, -43.12],
        [172.68, -34.63],
        [167.75, -42.28],
        [166.68, -14.63],
        [144.68, 13.25],
        [146.93, -6.96],
        [141.01, 0.09]])

    p = torch.Tensor(p.reshape([1, p.shape[0], 2]))
    v = torch.Tensor(v.reshape([1, v.shape[0], 2]))

    T = TPS.solve_system(p, v)
    point = np.array([event.xdata, event.ydata])
    point_T = TPS.point_transform(point, T, p)
    print("Longitude:", point_T[0, 0, 0])
    print("Latitude:", point_T[0, 1, 0])

if __name__ == '__main__':
    print("It is suggested that clicking on the image close to the middle position will be more accurate.")
    fig = plt.figure()
    img = Image.open('2048x2048.jpg')
    plt.imshow(img, animated= True)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()
