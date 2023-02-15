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
df = df[(df['date'] >= '2020-06-01') & (df['date'] <= '2021-02-01')]

# Focussed dates for trends
# df = df[(df['date'] >= '2020-08-01') & (df['date'] <= '2020-10-01')]


df_cali = df[(df['region'] == 'CA')]

df_az = df[(df['region'] == 'AZ')]

df_oregon = df[(df['region'] == 'OR')]

df_nevada = df[(df['region'] == 'NV')]

states =['California', 'Arizona', 'Oregon', 'Nevada']



for ind, frame in enumerate([df_cali, df_az, df_oregon, df_nevada]):
        plt.plot(frame['date'], frame['all_mobility'], label=states[ind])

plt.legend()
plt.xlabel("Time")
plt.ylabel("% Change in Mobility")
plt.show()