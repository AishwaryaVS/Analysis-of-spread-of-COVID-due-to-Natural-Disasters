import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator


df = pd.read_csv('merged-sets-alex.csv')
df['date'] = pd.to_datetime(df.date)

df['all_mobility'] = df['retail_and_recreation_percent_change_from_baseline'] + df['grocery_and_pharmacy_percent_change_from_baseline'] + df['parks_percent_change_from_baseline'] + df['transit_stations_percent_change_from_baseline'] + df['workplaces_percent_change_from_baseline'] + df['residential_percent_change_from_baseline'] + df['apple_mobility']

# Overall trend dates
df = df[(df['date'] >= '2021-08-01') & (df['date'] <= '2021-10-01')]

print('DF: \n', df)
# Focussed dates for trends
# df = df[(df['date'] >= '2020-08-01') & (df['date'] <= '2020-10-01')]


df_louisiana = df[(df['region'] == 'LA')]

df_tx = df[(df['region'] == 'TX')]

df_ak = df[(df['region'] == 'AR')]

df_miss = df[(df['region'] == 'MS')]

states =['Louisiana', 'Arkansas', 'Mississippi', 'Texas']



for ind, frame in enumerate([df_louisiana, df_ak, df_miss, df_tx]):
        plt.plot(frame['date'], frame['all_mobility'], label=states[ind])

plt.legend()
plt.xlabel("Time")
plt.ylabel("% Change in Mobility")
plt.show()