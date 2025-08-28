from datetime import datetime, timezone, timedelta

def check_trading_time():
    now = datetime.now(timezone.utc)
    trading_time_start = now.replace(hour=12, minute=2, second=0, microsecond=0)
    trading_time_end = trading_time_start + timedelta(hours=4, minutes=10)
    return trading_time_start <= now < trading_time_end