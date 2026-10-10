import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# 1. 加载数据，使用 NumPy 数组
iris = load_iris()
X = iris.data
y = iris.target

# 2. 统计每种花的数量和平均尺寸
print("\n各花种的数据统计：")

for label in range(3):
    flowers = X[y == label]

    print("\n花种：", iris.target_names[label])
    print("样本数量：", len(flowers))

    # axis=0：按列计算四个特征各自的平均值
    averages = flowers.mean(axis=0)

    for name, average in zip(iris.feature_names, averages):
        print(f"  mean {name}: {average:.2f}")

# 3. 绘制花瓣和花萼尺寸散点图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 每张图使用的横轴列、纵轴列和标题
plot_settings = [
    (2, 3, "Iris Petal Measurements"),
    (0, 1, "Iris Sepal Measurements")
]

for ax, (x_col, y_col, title) in zip(axes, plot_settings):
    for label in range(3):
        flowers = X[y == label]

        ax.scatter(
            flowers[:, x_col],
            flowers[:, y_col],
            label=iris.target_names[label]
        )

    ax.set_xlabel(iris.feature_names[x_col])
    ax.set_ylabel(iris.feature_names[y_col])
    ax.set_title(title)
    ax.legend()

fig.tight_layout()

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. 创建模型：先标准化，再使用 KNN 分类
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

# 标准化参数只从训练集学习
model.fit(X_train, y_train)

# 6. 预测并计算准确率
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
wrong_count = (predictions != y_test).sum()

print("\n训练样本数量：", len(X_train))
print("测试样本数量：", len(X_test))
print(f"测试准确率：{accuracy:.2%}")
print("预测错误数量：", wrong_count)

# 7. 展示前 10 个测试样本的预测结果
print("\n前 10 个测试样本的预测结果：")

for true_label, predicted_label in zip(y_test[:10], predictions[:10]):
    true_name = iris.target_names[true_label]
    predicted_name = iris.target_names[predicted_label]
    print(f"真实：{true_name} | 预测：{predicted_name}")

# 8. 输出分类错误的样本
if wrong_count > 0:
    print("\n分类错误的样本：")

    for i in range(len(y_test)):
        if predictions[i] != y_test[i]:
            print("测量值：", X_test[i])
            print("真实花种：", iris.target_names[y_test[i]])
            print("预测花种：", iris.target_names[predictions[i]])
            print()
else:
    print("\n本次测试集全部预测正确。")

# 9. 绘制混淆矩阵
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1, 2],
    display_labels=iris.target_names,
    cmap="Purples"
)

plt.title("Iris Classification Results")
plt.tight_layout()

# 显示散点图和混淆矩阵
plt.show()