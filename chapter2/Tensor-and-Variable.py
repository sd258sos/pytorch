import torch
import numpy as np
# 创建一个 numpy ndarray
numpy_tensor = np.random.randn(10, 20)
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)

# 如果 pytorch tensor 在 cpu 上
numpy_array = pytorch_tensor1.numpy()

# 如果 pytorch tensor 在 gpu 上
numpy_array = pytorch_tensor1.cpu().numpy()
# 第一种方式是定义 cuda 数据类型
dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)

# 第二种方式更简单，推荐使用
gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
# gpu_tensor = torch.randn(10, 20).cuda(1) # 将 tensor 放到第二个 GPU 上

cpu_tensor = gpu_tensor.cpu()

# 可以通过下面两种方式得到 tensor 的大小
print("获取 tensor 的大小")
print(pytorch_tensor1.shape)
print(pytorch_tensor1.size())

# 得到 tensor 的数据类型
print("得到 tensor 的数据类型")
print(pytorch_tensor1.type())

# 得到 tensor 的维度
print("得到 tensor 的维度")
print(pytorch_tensor1.dim())

# 得到 tensor 的所有元素个数
print("得到 tensor 的所有元素个数")
print(pytorch_tensor1.numel())

# 小练习
#
# 查阅以下文档了解 tensor 的数据类型，创建一个 float64、大小是 3 x 2、随机初始化的 tensor，将其转化为 numpy 的 ndarray，输出其数据类型
#
# 参考输出: float64
print("答案1")
x = torch.randn(3, 2)
x = x.type(torch.DoubleTensor)
x_array = x.numpy()
print(x_array.dtype)

x = torch.ones(2, 2)
print(x) # 这是一个float tensor

print(x.type())

# 将其转化为整形
print("将其转化为整形")
x = x.long()
# x = x.type(torch.LongTensor)
print(x)

# 再将其转回 float
print("再将其转回 float")
x = x.float()
# x = x.type(torch.FloatTensor)
print(x)

x = torch.randn(4, 3)
print(x)

# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)

# 每一行的最大值
max_value

# 每一行最大值的下标
max_idx

# 沿着行对 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)

# 增加维度或者减少维度
print(x.shape)
x = x.unsqueeze(0) # 在第一维增加
print(x.shape)

x = x.unsqueeze(1) # 在第二维增加
print(x.shape)

x = x.squeeze(0) # 减少第一维
print(x.shape)

x = x.squeeze() # 将 tensor 中所有的一维全部都去掉
print(x.shape)

x = torch.randn(3, 4, 5)
print(x.shape)

# 使用permute和transpose进行维度交换
x = x.permute(1, 0, 2) # permute 可以重新排列 tensor 的维度
print(x.shape)

x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
print(x.shape)

# 使用 view 对 tensor 进行 reshape
x = torch.randn(3, 4, 5)
print(x.shape)

x = x.view(-1, 5) # -1 表示任意的大小，5 表示第二维变成 5
print(x.shape)

x = x.view(3, 20) # 重新 reshape 成 (3, 20) 的大小
print(x.shape)

x = torch.randn(3, 4)
y = torch.randn(3, 4)

# 两个 tensor 求和
z = x + y
# z = torch.add(x, y)

x = torch.ones(3, 3)
print(x.shape)

# unsqueeze 进行 inplace
x.unsqueeze_(0)
print(x.shape)

# transpose 进行 inplace
x.transpose_(1, 0)
print(x.shape)

x = torch.ones(3, 3)
y = torch.ones(3, 3)
print(x)

# add 进行 inplace
x.add_(y)
print(x)


# 答案
x = torch.ones(4, 4).float()
x[1:3, 1:3] = 2
print(x)

