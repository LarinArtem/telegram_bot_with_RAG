from eodhd import APIClient
import requests
import pandas as pd



EOD_API_TOKEN = "YOUR_API_TOKEN"


def undervalued_stocks():
    client = APIClient(EOD_API_TOKEN)
    filters = '[["market_capitalization",">",5000000000],["market_capitalization","<",25000000000],["exchange","=","NASDAQ","NYSE"]]'
    results = client.stock_market_screener(filters=filters, sort='market_capitalization.desc', limit=10, signals="wallstreet_lo")
    df_results = pd.DataFrame(results['data'])
    df_results = df_results[['code','industry','adjusted_close']]
    response = ""
    for index, row in df_results.iterrows():
        code = row['code']
        industry = row['industry']
        undervalue = row['adjusted_close']
        response += f"{code} - {industry} - ${undervalue}\n"
    return response