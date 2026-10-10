# 鸢尾花数据分析与分类

使用 Python 分析 Iris 数据集，并通过 KNN 模型预测鸢尾花种类。

## 技术栈

Python、NumPy、Matplotlib、scikit-learn

## 安装与运行

安装依赖：

    python -m pip install scikit-learn matplotlib

运行程序：

    python iris_analysis.py

## 项目功能

- 统计各花种的样本数量和平均尺寸
- 绘制花瓣、花萼尺寸散点图
- 使用标准化与 KNN 进行分类
- 输出准确率和预测错误的样本
- 绘制混淆矩阵

## 数据与模型

- 数据：scikit-learn 自带的 Iris 数据集
- 样本：150 个，包含 3 种鸢尾花
- 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
- 划分：80% 训练集、20% 测试集
- 模型：StandardScaler + KNeighborsClassifier(n_neighbors=5)
- 随机种子：random_state=42

## 结果说明

准确率以实际运行输出为准。本项目根据测量数据分类，不支持直接识别花的照片。

## 参考

[scikit-learn Iris 数据集](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)