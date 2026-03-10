import torch
# 打印PyTorch版本
print(f"PyTorch 版本: {torch.__version__}")
# 检查CUDA是否可用（CPU版显示False，GPU版显示True）
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
# 创建5行3列的随机张量
x = torch.rand(5, 3)
print("随机张量x:\n", x)