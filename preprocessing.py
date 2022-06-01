import numpy as np
import pandas as pd
import sklearn
import plotly
import tensorflow
import xgboost as xgb
import glmnet_python

TLV_STRING = 'תל אביב - יפו'

def preprocess(df: pd.DataFrame):
    """gets RAW df and producues data ready for learning."""
    df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
    df = df.sort_values(by=['update_date'])
    df_tlv = df['ci']
    pass


if __name__ == '__main__':
    file_path = "Mission 1 - Waze/waze_data.csv"
    df = pd.read_csv(file_path, parse_dates=['pubDate', 'update_date'])
    processed_df = preprocess(df)
