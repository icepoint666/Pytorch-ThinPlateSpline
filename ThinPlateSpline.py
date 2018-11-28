import torch
import numpy as np
import torch.nn.functional as F

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def solve_system(coord, vec):
    """Thin Plate Spline Spatial Transformer layer
    TPS control points are arranged in arbitrary positions given by `coord`.
    coord : float Tensor [num_batch, num_point, 2]
        Relative coordinate of the control points.
    vec : float Tensor [num_batch, num_point, 2]
        The vector on the control points.
    """
    num_batch = coord.shape[0]
    num_point = coord.shape[1]

    ones = torch.ones([num_batch, num_point, 1])
    p = torch.cat([ones, coord], 2)     # [bn, pn, 3]
    p_1 = torch.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
    p_2 = torch.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
    d = p_1 - p_2                                  # [bn, pn, pn, 3]
    d2 = torch.sum(torch.pow(d, 2), 3)  # [bn, pn, pn]
    r = d2 * torch.log(d2 + 1e-6)       # [bn, pn, pn]

    zeros = torch.zeros([num_batch, 3, 3])
    W_0 = torch.cat([p, r], 2)          # [bn, pn, 3+pn]
    W_1 = torch.cat([zeros, torch.transpose(p, 2, 1)], 2)  # [bn, 3, pn+3]
    W = torch.cat([W_0, W_1], 1)        # [bn, pn+3, pn+3]
    W_inv = b_inv(W)

    tp = F.pad(vec.unsqueeze(1), (0, 0, 0, 3))
    tp = tp.squeeze(1)                  # [bn, pn+3, 2]
    T = torch.matmul(W_inv, tp)         # [bn, pn+3, 2]
    T = torch.transpose(T, 2, 1)        # [bn, 2, pn+3]

    return T

def point_transform(point, T, coord):
    point = torch.Tensor(point.reshape([1, 1, 2]))
    d2 = torch.sum(torch.pow(point - coord, 2), 2)
    r = d2 * torch.log(d2 + 1e-6)
    q = torch.Tensor(np.array([[1, point[0, 0, 0], point[0, 0, 1]]]))
    x = torch.cat([q, r], 1)
    point_T = torch.matmul(T, torch.transpose(x.unsqueeze(1), 2, 1))
    return point_T

