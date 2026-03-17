import torch
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义目标函数：f(x) = x² + 2x + 1
def target_fun(x):
    return x**2 + 2*x + 1

# ===================== 核心优化过程 =====================
x = torch.tensor([5.0], requires_grad=True)  # 初始值x=5
lr = 0.1  # 学习率
epochs = 20  # 迭代次数
x_history = [x.item()]  # 记录x的变化
loss_history = [target_fun(x).item()]  # 记录损失的变化

print("="*30)
print("梯度下降参数收敛过程")
print("="*30)
for epoch in range(epochs):
    y = target_fun(x)
    y.backward()  # 计算梯度
    # 更新参数
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()  # 清空梯度
    # 记录数据
    x_history.append(x.item())
    loss_history.append(target_fun(x).item())
    # 每5轮打印一次
    if (epoch+1) % 5 == 0:
        print(f"第{epoch+1}轮 | x = {x.item():.6f} | 损失 = {target_fun(x).item():.6f}")
# 最终结果
print(f"\n迭代结束 | 最优x ≈ {x.item():.6f} | 最小损失 ≈ {target_fun(x).item():.6f}")
print("理论最优解：x=-1，损失=0")
print("="*30)

# ===================== 可视化部分 =====================
plt.figure(figsize=(10, 4))
# 子图1：损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(epochs+1), loss_history, marker='o', color='red', linewidth=2, markersize=4)
plt.xlabel("迭代次数 (Epoch)", fontsize=12)
plt.ylabel("损失值 (Loss)", fontsize=12)
plt.title("梯度下降损失曲线", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='理论最小损失')
plt.legend()

# 子图2：优化路径
x_vals = np.linspace(-6, 6, 200)
y_vals = target_fun(torch.tensor(x_vals)).numpy()
plt.subplot(1, 2, 2)
plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='目标函数 f(x)=x²+2x+1')
plt.scatter(x_history, loss_history, color='red', s=50, zorder=5, label='优化路径')
plt.axvline(x=-1, color='green', linestyle='--', linewidth=2, label='理论最优x=-1')
plt.xlabel("参数 x", fontsize=12)
plt.ylabel("f(x) (损失值)", fontsize=12)
plt.title("梯度下降优化路径", fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()