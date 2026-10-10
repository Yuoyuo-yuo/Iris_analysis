from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 加载鸢尾花数据集
    iris = load_iris()
    print('鸢尾花数据集', iris)
    print('鸢尾花描述', iris.DESCR)
    print("特征数据值", iris.data)
    print("特征名称", iris.feature_names)
    print("特征数据形状", iris.data.shape)
    print("目标值名称", iris.target_names)
    print("目标值", iris.target)


