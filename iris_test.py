from sklearn.datasets import load_digits
digits=load_digits()
X, y = digits.data, digits.target
print(digits.images[:10],X[:10],y[:10])
print("第一张图片的像素：")
print(digits.images[0])

print("展开后的像素：")
print(X[0])

print("正确答案：", y[0])
