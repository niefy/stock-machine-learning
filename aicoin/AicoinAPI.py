from HttpUtil import httpGet, httpPost

API_BASE_URL = 'www.aicoin.net.cn'


def kline(symbol='huobiprobtcusdt', step=3600):
    K_LINE_RESOURCE = '/chart/api/data/period'
    params = 'symbol=%s&step=%d' % (symbol, step)
    result = httpGet(API_BASE_URL, K_LINE_RESOURCE,params)
    return result['data']
