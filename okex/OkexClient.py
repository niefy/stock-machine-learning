# coding=utf-8
# 客户端调用，用于查看API返回结果
from OkexSpotAPI import OkexSpot
import talib
import json
import numpy as np
import pandas as pd


# 初始化apikey，secretkey,url
apikey = 'XXXX'
secretkey = 'XXXXX'
okexRESTURL = 'www.okex.com'
# 现货API
okexSpot = OkexSpot(okexRESTURL, apikey, secretkey)

print('K线数据')
# 方式1：从API接口获取最新K线数据
# kline_data_json = okexSpot.kline('btc_usdt')

# 方式2 使用本地保存的数据(不是最新)，避免接口连接缓慢或超时
kline_data_json = json.loads(open('kline_data.txt', 'r').read())


kline_data = pd.DataFrame(kline_data_json,
                          columns=['time', 'open', 'high', 'low', 'close', 'volume'])
print(kline_data)

# 提取数据列
close_prices = np.array(kline_data['close'], dtype=np.double)
high_prices = np.array(kline_data['high'], dtype=np.double)
low_prices = np.array(kline_data['low'], dtype=np.double)
volumes = np.array(kline_data['volume'], dtype=np.double)

#通过数据计算指标
ma7_data = talib.MA(close_prices,timeperiod=7)
ma30_data = talib.MA(close_prices,timeperiod=30)
# wma_data = talib.WMA(close_prices)
# mom_data = talib.MOM(close_prices)
# stck, stcd = talib.STOCH(high_prices, low_prices, close_prices)
macd, macdsignal, macdhist = talib.MACD(close_prices)
rsi_data = talib.RSI(close_prices,timeperiod=10)
boll_upper, boll_middle, boll_lower= talib.BBANDS(close_prices)
print('macd',macd)

# 保存训练数据中的一组训练数据
features = []
features.append(ma7_data)
features.append(macd)
features.append(macdsignal)
features.append(macdhist)
features.append(rsi_data)
features.append(boll_upper)
features.append(boll_middle)
features.append(boll_lower)