import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# ===================== 核心计算部分 =====================
# 1. 构造0-1样本数据
data = torch.tensor([1., 1., 0., 1., 0.])
# 2. 计算MLE
p_mle = torch.mean(data)
# 3. 计算MAP（Beta先验α=2，β=2）
alpha = 2
beta_prior = 2
sum_data = torch.sum(data)
N = len(data)
p_map = (sum_data + alpha - 1) / (N + alpha + beta_prior - 2)
# 4. 输出结果
print("="*30)
print("参数估计（MLE & MAP）结果")
print("="*30)
print(f"样本数据：{data.numpy()}")
print(f"MLE估计值 p = {p_mle.item():.4f}")
print(f"MAP估计值 p = {p_map.item():.4f}")
print("="*30)

# ===================== 可视化部分 =====================
# 生成p的取值区间（0-1，100个点）
p = np.linspace(0, 1, 100)
# 计算似然、先验、后验
likelihood = p ** sum_data.numpy() * (1 - p) ** (N - sum_data.numpy())
prior = beta.pdf(p, alpha, beta_prior)
posterior = beta.pdf(p, sum_data.numpy() + alpha, N - sum_data.numpy() + beta_prior)
# 绘图
plt.figure(figsize=(10, 6))
plt.plot(p, likelihood, label="Likelihood（似然）", color='blue', linewidth=2)
plt.plot(p, prior, label="Prior（Beta先验）", color='orange', linewidth=2)
plt.plot(p, posterior, label="Posterior（后验）", color='green', linewidth=2, linestyle='--')
plt.axvline(p_mle, color='red', linestyle='-.', label=f"MLE = {p_mle:.4f}", linewidth=2)
plt.axvline(p_map, color='purple', linestyle=':', label=f"MAP = {p_map:.4f}", linewidth=2)
plt.xlabel("参数 p (P(x=1))", fontsize=12)
plt.ylabel("概率密度/似然值", fontsize=12)
plt.title("MLE vs MAP 对比", fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()