# coding=utf-8
# 客户端调用，用于查看API返回结果

from OkexSpotAPI import OkexSpot

# 初始化apikey，secretkey,url
apikey = 'XXXX'
secretkey = 'XXXXX'
okexRESTURL = 'www.okex.com'
# 现货API
okexSpot = OkexSpot(okexRESTURL, apikey, secretkey)

print('K线数据')
kline_data = okexSpot.kline('btc_usdt')
print(kline_data)