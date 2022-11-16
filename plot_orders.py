import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

f = './robotex5.csv'
time_col = 'start_time'
df = pd.read_csv(f, parse_dates=[time_col])
df = df.sort_values(by=[time_col]).reset_index(drop=True)
# group orders by hour each day
df['hour'] = df[time_col].dt.hour
df['date'] = df[time_col].dt.date
df['time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
# df = df[:20000]
orders_count = df.groupby(['time']).size()
ax = orders_count.plot(
    kind='bar',
    width=1,
    xticks=orders_count.index.hour,
    xlabel='Days',
    ylabel='Order count'
)

# from matplotlib.dates import AutoDateFormatter, AutoDateLocator

# xtick_locator = AutoDateLocator()
# xtick_formatter = AutoDateFormatter(xtick_locator)

# ax.xaxis.set_major_locator(xtick_locator)
# ax.xaxis.set_major_formatter(xtick_formatter)

# ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

plt.xticks(rotation=45)
# plt.savefig('one_day.png')
plt.show()
