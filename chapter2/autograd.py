import torch
from torch.autograd import Variable
x = Variable(torch.Tensor([2]), requires_grad=True)
y = x + 2
z = y ** 2 + 3
print(z)

x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)
out=y - torch.matmul(x, w)
print(out)
out = torch.mean(y - torch.matmul(x, w)) # torch.matmul 是做矩阵乘法
out.backward()
# 得到 x 的梯度
# print(x.grad)
print(out)