import requests
import pandas as pd

def get_latest_rankings_df():
    url = "https://apewisdom.io/api/v1.0/filter/all/page/1"
    data = requests.get(url).json()
    data_df = pd.DataFrame(data)
    rankings = data_df['results']
    rankings_df = pd.json_normalize(rankings)
    return rankings_df

def format_rankings_table(df):
    df = df[['rank','ticker', 'mentions', 'upvotes', 'rank_24h_ago', 'mentions_24h_ago']]
    df = df[0:20]  # Limit to top 20
    df = df.rename(columns={
        'ticker': 'Ticker',
        'mentions': 'Mentions',
        'upvotes': 'Upvotes',
        'rank_24h_ago': 'Rank 24h Ago',
        'mentions_24h_ago': 'Mentions 24h Ago'
    })
    ranking_text = "No. | T | Refs | Up | No. 24h | Refs 24h\n"
    for index, row in df.iterrows():
        rank = row['rank']
        ticker = row['Ticker']
        mentions = row['Mentions']
        upvotes = row['Upvotes']
        rank_24h_ago = row['Rank 24h Ago']
        mentions_24h_ago = row['Mentions 24h Ago']
        percent_change_mentions =  ((mentions - mentions_24h_ago) / mentions_24h_ago) * 100 if mentions_24h_ago else 0
        rank_change_percent = ((rank_24h_ago - rank) / rank_24h_ago) * 100 if rank_24h_ago else 0
        ranking_text +=f"{rank}. {ticker} - {mentions} | {upvotes} | {rank_24h_ago} ({round(percent_change_mentions, 2)}%) | {mentions_24h_ago} ({round(rank_change_percent, 2)}%)\n"
    return ranking_text