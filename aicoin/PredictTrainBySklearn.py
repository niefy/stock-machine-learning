# 通过SVM训练模型，进行涨跌预测
from AicoinClient import AicoinClient
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

TEST_DATA_NUM = 50  # 测试数据数量

# 计算预测准确率
def get_prediction_rate(y_test, prediction):
    prediction_right = 0
    for i in range(TEST_DATA_NUM):
        if y_test[i] == prediction[i]:
            prediction_right = prediction_right + 1
    return prediction_right / prediction.size

aicoinClient = AicoinClient("huobiprobtcusdt", step=3600)
features = aicoinClient.get_kline_features()
x_data = features[49:-1]  # 前面含较多null值，最后一组值无涨跌结果，故去头去尾处理

close_prices = np.array(aicoinClient.kline_data['close'], dtype=np.double)
y_data = (close_prices[50:] - close_prices[49:-1]) > 0  # 后一周期的价格是否大于前一个周期的价格
print("x_data:", x_data.shape, "y_data:", y_data.shape)

# 分训练数据和测试数据
x_train, y_train = x_data[:-TEST_DATA_NUM], y_data[:-TEST_DATA_NUM]  # 留部分组数据用于检验训练效果
x_test, y_test = x_data[-TEST_DATA_NUM:], y_data[-TEST_DATA_NUM:]
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:", x_test.shape, "y_test:", y_test.shape)

# LinearSVC
linear_svc = svm.LinearSVC()
linear_svc.fit(x_train, y_train)
prediction = linear_svc.predict(x_test)
print("LinearSVC准确率：", get_prediction_rate(y_test, prediction))

# KNeighborsClassifier
neighbors = KNeighborsClassifier()
neighbors.fit(x_train, y_train)
prediction = neighbors.predict(x_test)
print("KNeighborsClassifier准确率", get_prediction_rate(y_test, prediction))

# LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(x_train, y_train)
prediction = linearRegression.predict(x_test)>0.5
print("LinearRegression准确率",get_prediction_rate(y_test, prediction))