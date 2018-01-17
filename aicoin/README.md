# 使用[AICOIN](https://www.aicoin.net.cn/)的数据进行数字货币K线趋势研究
	在短周期内K线的变化更多受随机交易影响，故此进行短期趋势预测用于指导断线交易
	>- 使用tablib库进行指标计算
	>- 使用sklearn进行训练，模拟特征值
	
# AICOIN API
 - AICOIN未给出接口文档，可根据网页的网络请求来解析接口地址
 - AICOIN接口根地址:www.aicoin.net.cn
 - 报价接口
     - BTC-USDT价格：/chart/api/data/period?symbol=huobiprobtcusdt&step=3600