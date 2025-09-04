from eodhd import APIClient
import requests
import pandas as pd



EOD_API_TOKEN = "YOUR_API_TOKEN"



def momentum_stocks():
    client = APIClient(EOD_API_TOKEN)
    filters = '[["market_capitalization",">",5000000000],["market_capitalization","<",25000000000],["exchange","=","NASDAQ","NYSE"],["refund_5d_p",">",10]]'
    results = client.stock_market_screener(filters=filters, sort='market_capitalization.desc', limit=20)
    df_results = pd.DataFrame(results['data'])
    df_results = df_results[['code','industry','refund_5d_p']]
    response = ""
    for index, row in df_results.iterrows():
        code = row['code']
        momentum = row['refund_5d_p']
        industry = row['industry']
        response += f"{code} - {industry} - {momentum}%\n"
    return response