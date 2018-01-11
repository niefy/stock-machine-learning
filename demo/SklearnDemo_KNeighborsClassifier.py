# DEMO from: https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-2-general-pattern/
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Iris是Sklearn 本身的数据库之一，可以用来练习，这种花有四个属性，花瓣的长宽，茎的长宽，根据这些属性把花分为三类
iris = datasets.load_iris()
iris_X = iris.data # 属性集合（ 花瓣的长宽，茎的长宽）
iris_y = iris.target # 结果集（花的分类：0、1、2代表不同的分类）
print("iris_X", iris_X)
print("iris_y", iris_y)

# 将70%数据作为训练数据，30%数据作为测试检验数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# 进行训练
knn = KNeighborsClassifier()
knn.fit(X_train, y_train) # 用 fit 来训练 training data，knn 就已经是训练好的模型


print("预测值",knn.predict(X_test))
print("实际值",y_test)
