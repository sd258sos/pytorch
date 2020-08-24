import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
# 打印 format的全部内容
print('a is:{}'.format(a))
print('a size is {}'.format(a.size()))
