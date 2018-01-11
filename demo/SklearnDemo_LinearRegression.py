# DEMO from: https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-3-database/
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Sklearn 的boston 房价数据
loaded_data = datasets.load_boston()
data_X = loaded_data.data # 属性集合
data_y = loaded_data.target # 结果集
print("data_X", data_X)
X = data_X[:,0]
print("X",X)
print("data_y", data_y)

# 定义模型
model = LinearRegression()
model.fit(data_X,data_y)

#使用前4组数据来预测，对比真实结果
print("预测值", model.predict(data_X[:4,:]))
print("真实值",data_y[:4])