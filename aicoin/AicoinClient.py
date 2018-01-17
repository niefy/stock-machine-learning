# 获取K线数据，计算特征值
import AicoinAPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

kline_data_json = AicoinAPI.kline("huobiprobtcusdt",step=3600)
kline_data = pd.DataFrame(kline_data_json,
                          columns=['time', 'open', 'high', 'low', 'close', 'volume'])
print('K线数据获取成功，kline_data:',kline_data.shape)
print('计算K线指标')
# 提取数据列
time = np.array(kline_data['time'], dtype=np.string_)
close_prices = np.array(kline_data['close'], dtype=np.double)
high_prices = np.array(kline_data['high'], dtype=np.double)
low_prices = np.array(kline_data['low'], dtype=np.double)
volumes = np.array(kline_data['volume'], dtype=np.double)

# 通过数据计算指标
ma7_data = talib.MA(close_prices, timeperiod=7)
ma30_data = talib.MA(close_prices, timeperiod=30)
# wma_data = talib.WMA(close_prices)
# mom_data = talib.MOM(close_prices)
# stck, stcd = talib.STOCH(high_prices, low_prices, close_prices)
macd, macdsignal, macdhist = talib.MACD(close_prices)
rsi_data = talib.RSI(close_prices, timeperiod=10)
boll_upper, boll_middle, boll_lower = talib.BBANDS(close_prices)
print("指标计算完成")

plt.xlabel('time')
plt.ylabel('price')
plt.plot(time, ma7_data,'r', label='ma7_data')
plt.plot(time, ma30_data,'b', label='ma30_data')
plt.legend(bbox_to_anchor=[0.3, 1])# 图例
plt.show()