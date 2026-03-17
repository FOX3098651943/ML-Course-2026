import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# 固定随机种子，结果可复现
np.random.seed(42)
# 1. 构造二维分类数据
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])
# 2. 生成网格点（用于绘制分类边界）
x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                     np.linspace(x2_min, x2_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
# 3. 颜色配置
cmap_light = ListedColormap(['#FFEEEE', '#EEFFEE'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
# 4. 遍历不同K值绘图
plt.figure(figsize=(15, 4))
K_list = [1, 3, 5]
for idx, k in enumerate(K_list):
    # 初始化并训练KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_model.fit(X, y)
    # 预测网格点标签
    Z = knn_model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    # 绘制子图
    plt.subplot(1, 3, idx+1)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, s=100, edgecolor='black')
    plt.title(f"KNN Classification (K = {k})", fontsize=12, fontweight='bold')
    plt.xlabel("Feature 1", fontsize=10)
    plt.ylabel("Feature 2", fontsize=10)
    plt.grid(True, alpha=0.3)
# 显示图表
plt.tight_layout()
plt.show()
# 5. 输出测试集准确率
X_test = np.array([[2,2], [7,6], [4,4], [5,5]])
y_test = np.array([0,1,0,1])
print("="*30)
print("KNN分类准确率结果")
print("="*30)
for k in K_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    acc = model.score(X_test, y_test)
    print(f"K={k} 时，测试集准确率：{acc*100:.0f}%")
print("="*30)