# 获取K线数据，计算特征值
import AicoinAPI
import numpy as np
import pandas as pd
import talib

class AicoinClient:
    def __init__(self, symbol, step=3600):
        self.__symbol = symbol
        self.__step = step
        kline_data_json = AicoinAPI.kline(symbol, step=step)
        # K线数据值，包含[时间,开,高,低,收,量]
        self.kline_data = pd.DataFrame(kline_data_json, columns=['time', 'open', 'high', 'low', 'close', 'volume'])


    # 计算当前的K线特征值
    def get_kline_features(self):
        kline_data = self.kline_data
        # 提取数据列
        close_prices = np.array(kline_data['close'], dtype=np.double)
        high_prices = np.array(kline_data['high'], dtype=np.double)
        low_prices = np.array(kline_data['low'], dtype=np.double)
        volumes = np.array(kline_data['volume'], dtype=np.double)

        # 计算股票指标
        MA = talib.MA(close_prices, timeperiod=7)
        EMA = talib.MA(close_prices, timeperiod=7, matype=1)
        MACD, MACD_SIGNAL, MACD_HIST = talib.MACD(close_prices)
        RSI = talib.RSI(close_prices, timeperiod=10)
        BOLL_UPPER, BOLL_MIDDLE, BOLL_LOWER = talib.BBANDS(close_prices)
        SAR = talib.SAR(high_prices, low_prices)
        KDJ_K, KDJ_D = talib.STOCH(high_prices, low_prices, close_prices)

        # 合并组成特征数据
        features = np.vstack((volumes, MA, EMA, RSI, BOLL_UPPER, BOLL_MIDDLE, BOLL_LOWER, SAR,
                              MACD, MACD_SIGNAL, MACD_HIST, KDJ_K, KDJ_D)).T
        print("特征数据features.shape:", features.shape)
        return features

# TEST
# plt.xlabel('time')
# plt.ylabel('price')
# plt.plot(time, ma7_data, label='ma7_data')
# plt.plot(time, ma30_data, label='ma30_data')
# plt.legend(bbox_to_anchor=[0.3, 1])  # 图例
# plt.show()
