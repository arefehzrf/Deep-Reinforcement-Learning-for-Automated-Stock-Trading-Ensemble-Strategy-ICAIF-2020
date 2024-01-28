import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    # Load the dataset from CSV
    _data = pd.read_csv(file_name)

    if add_gold_data:
        if not start_date or not end_date:
            raise ValueError("Start date and end date must be provided to add gold data.")

        # Fetch gold price data
        gold_data = yf.download('GC=F', start=start_date, end=end_date)['Close']
        gold_data = gold_data.rename('gold_price')

        # Aligning the gold data date index with your DataFrame
        # Assuming your DataFrame has a 'date' column
        _data['date'] = pd.to_datetime(_data['date'])
        gold_data.index = pd.to_datetime(gold_data.index)
        gold_data = gold_data.reindex(_data['date'], method='ffill')  # Forward fill for missing dates

        # Merge the gold data into your existing DataFrame
        _data = _data.join(gold_data, on='date')
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    vix_data = yf.download('^VIX', start=df.iloc[0][0], end=df.iloc[-1][0])['Close']
    turbulence_index = pd.DataFrame(vix_data)
    return turbulence_index










