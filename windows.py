from config import WINDOWS

def filter_window(df, start_date):
    return df[df["date"] >= start_date]
