import pandas as pd
import plotly.express as px


def show_event_map():
    file_path = "datasets/original_data_cleaned.csv"
    df = pd.read_csv(file_path, parse_dates=['pubDate', 'update_date'])
    df['linqmap_city'] = df['linqmap_city'].astype(str)
    fig = px.scatter(df, x='x', y='y', color='linqmap_city')
    fig.show()


def show_hour_histogram():
    file_path = "datasets/original_data_cleaned.csv"
    df = pd.read_csv(file_path, parse_dates=['pubDate'])
    df = df[df['linqmap_city'] == 'תל אביב - יפו']
    hours_df = pd.DataFrame(pd.to_datetime(df['pubDate']).dt.hour)
    df = hours_df.value_counts().sort_index()
    fdf = df.reset_index()
    fig = px.histogram(fdf, x='pubDate', y=0, nbins=24)
    fig.show()


if __name__ == "__main__":
    show_event_map()
    show_hour_histogram()
