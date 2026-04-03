import pandas as pd
import pandas_market_calendars as mcal

nyse = mcal.get_calendar("NYSE")

def get_next_trading_day():
    today = pd.Timestamp.today().normalize()

    schedule = nyse.schedule(start_date=today, end_date=today + pd.Timedelta(days=10))
    future_days = schedule.index

    next_day = future_days[future_days > today][0]
    return next_day.strftime("%Y-%m-%d")
