# 正规化（标准化）数据，可提升机器学习的成效
# DEMO from: https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-1-normalization/
from sklearn import preprocessing
# 将资料分割为train和test的模块
from sklearn.model_selection import train_test_split
# 生成适合做分类资料的模块
from sklearn.datasets.samples_generator import make_classification
# SVC分类器模块
from sklearn.svm import SVC
# 可视化模块
import matplotlib.pyplot as plt

# 生成具有两种属性的300条数据
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
# 可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# 将数据分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train,y_train)
print("标准化前准确率：",svc.score(X_test,y_test))

X_new = preprocessing.scale(X)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.3)
svc = SVC()
svc.fit(X_train_new,y_train_new)
print("标准化后准确率：",svc.score(X_test_new,y_test_new))