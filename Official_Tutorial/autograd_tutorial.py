import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# Gradients
# out contains a single scalar
out.backward()

# Print gradients d(out)/dx
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# Now in this case y is no longer a scalar.
# torch.autograd could not compute the full Jacobian directly,
# but if we just want the vector-Jacobian product,
# simply pass the vector to backward as argument
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# Stop autograd from tracking history on Tensors with .requires_grad=True
# either by wrapping the code block in with torch.no_grad()
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# Or by using .detach() to get a new Tensor with the same content
# but that does not require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

"""
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
<AddBackward0 object at 0x00000184FA15C348>
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
False
True
<SumBackward0 object at 0x00000184FA37CA08>
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
tensor([ 246.7232, 1043.5968, 1234.8073], grad_fn=<MulBackward0>)
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
True
True
False
True
False
tensor(True)

Process finished with exit code 0
"""
