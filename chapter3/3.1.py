import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# 3.1.1 Tensor(张量)
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
# 打印 tensor的全部内容
print('a is:{}'.format(a))
# 打印 tensor的形状
print('a size is {}'.format(a.size()))

b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print('b is :{}'.format(b))
# 全部为0
c = torch.zeros((3, 2))
print('zero tensor: {}'.format(c))
# 正态分布作为随机初始值
d = torch.randn((3, 2))
print('normal random is : {}'.format(d))

a[0, 1] = 100
print('changed a is: {}'.format(a))

numpy_b = b.numpy()
print('conver to numpy is \n {}'.format(numpy_b))

e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()
print('change data type to float tensor: {}'.format(f_torch_e))

if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)

# 3.1.2 Variable(变量)

# Create Variable
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# Build a computational graph
y = w * x +b

# Compute gradients
y.backward()
print('after y backward!')
# Print out the gradients.
print(x.grad)
print(w.grad)
print(b.grad)

# 矩阵求导
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
print(y)

y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)