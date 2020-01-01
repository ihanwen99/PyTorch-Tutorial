from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size

print(x.size())

# Operations
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

# Use standard NumPy-like indexing with all bells and whistles
print(x[:, 1])

# Resizing: torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# One element tensor, use .item() to get the value
x = torch.randn(1)
print(x)
print(x.item())

# NumPy Bridge
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!

"""
tensor([[-7.7465e+09,  5.8714e-43, -7.7465e+09],
        [ 5.8714e-43, -7.7465e+09,  5.8714e-43],
        [-7.7465e+09,  5.8714e-43, -7.7466e+09],
        [ 5.8714e-43, -7.7466e+09,  5.8714e-43],
        [-7.7471e+09,  5.8714e-43, -7.7471e+09]])
tensor([[0.0905, 0.7857, 0.6172],
        [0.8006, 0.5651, 0.3790],
        [0.8637, 0.8435, 0.2057],
        [0.0174, 0.2721, 0.3276],
        [0.6892, 0.8417, 0.2257]])
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
tensor([5.5000, 3.0000])
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-1.6257,  1.5888, -1.1028],
        [ 1.1514, -0.1592, -0.2565],
        [ 0.2331, -0.3175,  0.4169],
        [ 0.1854,  1.5266,  1.3076],
        [-0.3950, -0.7177, -0.2146]])
torch.Size([5, 3])
tensor([[-1.1428,  1.7597, -0.2608],
        [ 2.0673,  0.4233,  0.6171],
        [ 0.3955,  0.2416,  1.2028],
        [ 0.9469,  1.9143,  2.0901],
        [ 0.1298,  0.2317,  0.2745]])
tensor([[-1.1428,  1.7597, -0.2608],
        [ 2.0673,  0.4233,  0.6171],
        [ 0.3955,  0.2416,  1.2028],
        [ 0.9469,  1.9143,  2.0901],
        [ 0.1298,  0.2317,  0.2745]])
tensor([[-1.1428,  1.7597, -0.2608],
        [ 2.0673,  0.4233,  0.6171],
        [ 0.3955,  0.2416,  1.2028],
        [ 0.9469,  1.9143,  2.0901],
        [ 0.1298,  0.2317,  0.2745]])
tensor([[-1.1428,  1.7597, -0.2608],
        [ 2.0673,  0.4233,  0.6171],
        [ 0.3955,  0.2416,  1.2028],
        [ 0.9469,  1.9143,  2.0901],
        [ 0.1298,  0.2317,  0.2745]])
tensor([ 1.5888, -0.1592, -0.3175,  1.5266, -0.7177])
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
tensor([1.1536])
1.1536121368408203
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
tensor([2.1536], device='cuda:0')
tensor([2.1536], dtype=torch.float64)

Process finished with exit code 0
"""