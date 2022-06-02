import plotly
import plotly.graph_objects as go
import pandas as pd


def show_event_map():
    TLV_STRING = ('תל אביב - יפו', 'TLV')
    file_path = "datasets/original_data.csv"
    df = pd.read_csv(file_path, parse_dates=['pubDate', 'update_date'])
    df_tlv = df[df['linqmap_city'] == TLV_STRING[0]].reset_index()

    xx, yy = df_tlv['x'], df_tlv['y']

    fig = go.Figure(
        frames=[go.Frame(data=[go.Scatter(x=[xx[k]], y=[yy[k]], mode="markers", marker=dict(color="red", size=10))])
                for k in range(df_tlv.shape[0])])
    fig.show()
