import torch
import numpy as np
import ThinPlateSpline as TPS

p = np.array([
  [0, 1],
  [-1, 0],
  [0, -1],
  [1, 0]])
v = np.array([
  [0, 0.75],
  [-1, 0.25],
  [0, -1.25],
  [1, 0.25]])

p = torch.Tensor(p.reshape([1, p.shape[0], 2]))
v = torch.Tensor(v.reshape([1, v.shape[0], 2]))

T = TPS.solve_system(p, v)

point = np.array([0, 0])
point_T = TPS.point_transform(point, T, p)
print(point_T)