# Main code for project
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.linear_model import OLS
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import os
import math
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def main():
    # extract NYT and Google datasets and turn them into our ARIMA datasets
    fire, flood, states = dataset()
    # import generated datasets
    firefiles = []
    floodfiles = []
    for s in fire:
        fireURL = 'fire_data/' + 'fire-' + s + ".csv"
        floodURL = 'flood_data/' + 'flood-' + s + ".csv"
        firefiles.append(fireURL)
        floodfiles.append(floodURL)
        fire[s].to_csv(fireURL)
        flood[s].to_csv(floodURL)
    arima(firefiles, floodfiles) # run ARIMA on the datasets.

    # plotting (results hardcoded here for simplicity)
    wildfire_results = {'AL': 510.86548207238724, 'AK': 58.36724662938837, 'AR': 303.44021141467954, 'AZ': 588.8481541479164, 'CA': 1692.7612819932956, 'CO': 394.38194365556916, 'CT': 681.4760224762191, 'DE': 66.62494201363829, 'DC': 26.23756000048049, 'FL': 1490.3561618819826, 'GA': 2194.5049333232378, 'ID': 206.6073334072225, 'IA': 634.5951519765841, 'IL': 2933.553650995288, 'IN': 654.8166879525505, 'KS': 1295.7366459062712, 'KY': 313.50241002014076, 'LA': 659.6011962581831, 'MA': 1084.765387456339, 'ME': 24.90102556148017, 'MS': 284.289381722815, 'MO': 831.831542073137, 'MT': 139.7979021833152, 'MD': 214.46867401917444, 'MI': 1973.5071807649374, 'MN': 663.3777573645169, 'NE': 358.512033378488, 'NC': 644.7674749633524, 'ND': 237.05179262414887, 'NH': 39.54388086316795, 'NJ': 785.2822636194068, 'NM': 166.05040703857543, 'NV': 213.32955869625374, 'NY': 1179.322595291901, 'OR': 94.4533290929897, 'OK': 574.3847613948267, 'OH': 567.3448911142884, 'PA': 832.2287694114617, 'RI': 321.51503032390747, 'SC': 453.2247259613907, 'SD': 243.19672220013283, 'TN': 930.8980801867249, 'TX': 2555.1266618589425, 'UT': 445.80560209432167, 'VA': 296.8532579194764, 'VT': 13.455323371820905, 'WA': 338.6401167214775, 'WI': 1134.6542507630209, 'WY': 160.17667956262045, 'WV': 99.33735341614702}
    hurricane_results = {'AL': 1768.4059823054972, 'AK': 658.4645983962818, 'AR': 714.8259157550418, 'AZ': 1523.290362629702, 'CA': 4450.101905552009, 'CO': 1747.7454091898194, 'CT': 499.5457119728351, 'DE': 158.99478905327015, 'DC': 159.0305846798858, 'FL': 8696.495290019726, 'GA': 4285.080007645244, 'ID': 724.8143439623361, 'IA': 1716.4436842042633, 'IL': 3217.826453433373, 'IN': 2585.830808503748, 'KS': 1347.9507073635652, 'KY': 2294.421058914559, 'LA': 1994.2338885976305, 'MA': 1707.6293318063597, 'ME': 364.078467774804, 'MS': 1294.5335688471503, 'MO': 1172.1306159232001, 'MT': 550.2175414291818, 'MD': 648.1069169053721, 'MI': 6490.36630068119, 'MN': 3020.7080181155243, 'NE': 654.3312013664231, 'NC': 4462.09429317206, 'ND': 236.47771369759766, 'NH': 805.6964468445884, 'NJ': 1072.8937478660594, 'NM': 1011.6712285044747, 'NV': 1112.9879274048033, 'NY': 3119.833451231984, 'OR': 1136.698446050502, 'OK': 1651.726966097971, 'OH': 3431.062027031242, 'PA': 3649.1541007760693, 'RI': 362.439459606152, 'SC': 3121.9737075225325, 'SD': 348.35433028132377, 'TN': 5022.155829928175, 'TX': 7769.497832416941, 'UT': 1039.6265058081456, 'VA': 2198.9760881078155, 'VT': 200.63710766587647, 'WA': 2103.765312232759, 'WI': 2547.422376670731, 'WY': 359.3406215110371, 'WV': 879.9872495625436}
    populations = {'AL': 5024279, 'AK': 733391, 'AR': 3011524, 'AZ': 7303398, 'CA':39538223, 'CO': 5773714, 'CT': 3605944, 'DE': 989948, 'DC': 689545, 'FL':  21538187, 'GA': 10711908, 'ID': 1839106, 'IA': 3190369, 'IL': 12812508, 'IN': 6785528, 'KS': 2937880, 'KY': 4505836, 'LA': 4657757, 'MA': 7029917, 'ME': 1362359, 'MS': 2961279, 'MO': 6154913, 'MT': 1084225, 'MD': 6177224, 'MI': 10077331, 'MN': 5706494, 'NE': 1961504, 'NC': 10439388, 'ND': 779094, 'NH': 1377529, 'NJ': 9288994, 'NM': 2117522, 'NV': 3104614, 'NY': 20201249, 'OR': 4237256, 'OK':  3959353, 'OH': 11799448, 'PA': 13002700, 'RI': 1097379, 'SC': 5118425, 'SD': 886667, 'TN': 6910840, 'TX': 29145505, 'UT': 3271616, 'VA':  8631393, 'VT': 643077, 'WA': 7705281, 'WI': 5893718, 'WY': 576851, 'WV': 1793716}
    wildfire_distances = {'AL': 1590, 'AK': 2243, 'AR': 1229, 'AZ': 161, 'CA': 0, 'CO': 573, 'CT': 2201, 'DE': 2095, 'DC': 2021, 'FL': 2030, 'GA': 1768, 'ID': 313, 'IA': 1198, 'IL': 1387, 'IN': 1525, 'KS': 1270, 'KY': 1629, 'LA': 1333, 'MA': 2247, 'ME': 2312, 'MS': 1424, 'MO': 1215, 'MT': 569, 'MD': 2028, 'MI': 1611, 'MN': 1244, 'NE': 949, 'NC': 1932, 'ND': 1017, 'NH': 2227, 'NJ': 2128, 'NM': 449, 'NV': 82, 'NY': 2079, 'OR': 177, 'OK': 971, 'OH': 1707, 'PA': 1982, 'RI': 2257, 'SC': 1908, 'SD': 987, 'TN': 1538, 'TX': 999, 'UT': 352, 'VA': 1976, 'VT': 2164, 'WA': 373, 'WI': 1392, 'WY': 641, 'WV': 1817}
    hurricane_distances = {'AL': 0, 'AK': 2749, 'AR': 130, 'AZ': 998, 'CA': 1450, 'CO': 736, 'CT': 1127, 'DE': 938, 'DC': 862, 'FL': 504, 'GA': 379, 'ID': 1276, 'IA': 621, 'IL': 519, 'IN': 530, 'KS': 408, 'KY': 480, 'LA': 0, 'MA': 1189, 'ME': 1337, 'MS': 0, 'MO': 377, 'MT': 1230, 'MD': 0, 'MI': 788, 'MN': 1244, 'NE': 603, 'NC': 645, 'ND': 1039, 'NH': 1218, 'NJ': 0, 'NM': 703, 'NV': 1299, 'NY': 0, 'OR': 1597, 'OK': 238, 'OH': 673, 'PA': 0, 'RI': 1177, 'SC': 546, 'SD': 824, 'TN': 311, 'TX': 222, 'UT': 1062, 'VA': 778, 'VT': 1192, 'WA': 1627, 'WI': 780, 'WY': 948, 'WV': 667}
    
    # Wildfire - correlation testing (scatter plot generation)
    epicenter = {}
    neighbors = {}
    far = {}
    for s in states:
        if s == 'CA':
            item = [s, wildfire_results[s], populations[s], wildfire_distances[s]]
            epicenter[s] = [wildfire_results[s], populations[s], wildfire_distances[s]]
            plt.scatter(item[2], item[1], c="red")
        elif s in ['NV', 'OR', 'AZ']:
            item = [s, wildfire_results[s], populations[s], wildfire_distances[s]]
            neighbors[s] = [wildfire_results[s], populations[s], wildfire_distances[s]]
            plt.scatter(item[2], item[1], c="yellow")
        else:
            item = [s, wildfire_results[s], populations[s], wildfire_distances[s]]
            far[s] = [wildfire_results[s], populations[s], wildfire_distances[s]]
            plt.scatter(item[2], item[1], c="green")
    plt.xlabel('Population')
    plt.ylabel('RMSE post-disaster')
    plt.title('California wildfires - Population vs error')
    plt.show()

    for s in epicenter:
        item = epicenter[s]
        plt.scatter(item[2], item[0], c="red")
    for s in neighbors:
        item = neighbors[s]
        plt.scatter(item[2], item[0], c="yellow")
    for s in far:
        item = far[s]
        plt.scatter(item[2], item[0], c="green")
    plt.xlabel('Distance to disaster epicenter')
    plt.ylabel('RMSE post-disaster')
    plt.title('California wildfires - Distance vs error')
    plt.show()

    # Hurricane (correlation testing, scatter plot generation)
    epicenter = {}
    neighbors = {}
    far = {}
    for s in states:
        if s in ['LA', 'NJ', 'NY', 'PA', 'MS', 'AL', 'MD', 'CT']:
            item = [s, hurricane_results[s], populations[s], hurricane_distances[s]]
            epicenter[s] = [hurricane_results[s], populations[s], hurricane_distances[s]]
            plt.scatter(item[2], item[1], c="red")
        elif s in ['TX', 'AR', 'TN', 'GA', 'FL', 'OH', 'WV', 'DC', 'VA', 'DE', 'VT', 'MA', 'RI']:
            item = [s, hurricane_results[s], populations[s], hurricane_distances[s]]
            neighbors[s] = [hurricane_results[s], populations[s], hurricane_distances[s]]
            plt.scatter(item[2], item[1], c="yellow")
        else:
            item = [s, hurricane_results[s], populations[s], hurricane_distances[s]]
            far[s] = [hurricane_results[s], populations[s], hurricane_distances[s]]
            plt.scatter(item[2], item[1], c="green")
    plt.xlabel('Population')
    plt.ylabel('RMSE post-disaster')
    plt.title('Hurricane Ida - Population vs error')
    plt.show()

    for s in epicenter:
        item = epicenter[s]
        plt.scatter(item[2], item[0], c="red")
    for s in neighbors:
        item = neighbors[s]
        plt.scatter(item[2], item[0], c="yellow")
    for s in far:
        item = far[s]
        plt.scatter(item[2], item[0], c="green")
    plt.xlabel('Distance to disaster epicenter')
    plt.ylabel('RMSE post-disaster')
    plt.title('Hurricane Ida - Distance vs error')
    plt.show()

#ARIMA model fitting to fire and flood datasets
@ignore_warnings(category=ConvergenceWarning)
def arima(a, b):
    # ['date', 'retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline','apple_mobility', 'distance', 'cases', 'deaths']
    #Wildfire
    fire_rmse = {}
    flood_rmse = {}
    #wildfire ARIMA fitting
    for f in a:
        data = pd.read_csv(f)
        exog_labels = ['mobility_sum','apple_mobility', 'distance']
        endog_labels = ['cases']
        data.fillna(0, inplace=True)
        train = data[:92]
        test = data[92:]
        # arima/ols order
        '''
        plot_pacf(train[['cases']].diff().dropna()) #p = 1
        #d = 1
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        plot_acf(train[['cases']], ax=ax1)
        plot_acf(train[['cases']].diff().dropna(), ax=ax2)
        plot_acf(train[['cases']].diff().dropna(), ax=ax3)
        plot_acf(train[['cases']].diff().dropna()) #q = 1
        plt.show()
        '''
        predictions = []
        exact = []
        for i in range(len(test)):
            model = ARIMA(train[endog_labels], train[exog_labels], order=(1,1,1), trend="t")
            #model = ARIMA(train[endog_labels], order=(1,1,1))
            model_fit = model.fit()
            #print(type(test.iloc[i][exog_labels]))
            exo = np.array(test.iloc[i][exog_labels], dtype=float)
            output = model_fit.forecast(exog=exo)
            #output = model_fit.forecast()
            #print(output)
            yhat = output[len(train)]
            predictions.append(yhat)
            obs = data.iloc[len(train)][['cases']]
            exact.append(obs)
            train = data[:93+i]
            #print('predicted=%f, expected=%f' % (yhat, obs))
        rmse = math.sqrt(mean_squared_error(exact, predictions))
        fire_rmse[f[-6:-4]] = rmse
        ''' removed due to inability to converge
        train = data[:92]
        test = data[92:]
        predictions = []
        exact = []
        for i in range(len(test)):
            #print(train[exog_labels])
            model = OLS(train[endog_labels], train[exog_labels], order=(1,1,1))
            model_fit = model.fit()
            output = model_fit.predict()
            yhat = output[0]
            predictions.append(yhat)
            obs = test.iloc[i][['cases']]
            #print('predicted=%f, expected=%f' % (yhat, obs))
            train = data[:93+i]
            exact.append(obs)
        '''
        #rmse = math.sqrt(mean_squared_error(exact, predictions))
        #print(rmse)
    #Hurricane
    #print(fire_rmse)

    # hurricane/flood ARIMA model

    for f in b:
        print(f[-6:-4])
        data = pd.read_csv(f)
        exog_labels = ['mobility_sum','apple_mobility', 'distance']
        endog_labels = ['cases']
        data.fillna(0, inplace=True)
        train = data[:92]
        test = data[92:]

        predictions = []
        exact = []
        for i in range(len(test)):
            model = ARIMA(train[endog_labels], train[exog_labels], order=(1,1,1), trend="t")
            #model = ARIMA(train[endog_labels], order=(1,1,1))
            model_fit = model.fit()
            #print(type(test.iloc[i][exog_labels]))
            exo = np.array(test.iloc[i][exog_labels], dtype=float)
            output = model_fit.forecast(exog=exo)
            #output = model_fit.forecast()
            #print(output)
            yhat = output[len(train)]
            predictions.append(yhat)
            obs = data.iloc[len(train)][['cases']]
            exact.append(obs)
            train = data[:93+i]
            #print('predicted=%f, expected=%f' % (yhat, obs))
        rmse = math.sqrt(mean_squared_error(exact, predictions))
        flood_rmse[f[-6:-4]] = rmse
    print(flood_rmse)
    return

   #generates dataaset for ARIMA 

def dataset():
    # import base datasets
    mobility = pd.read_csv('merged-sets-alex.csv') #1/21/20 - 11/3/22 - mobility data
    covid = pd.read_csv('us-states.csv') #1/1/20 - 1/8/22 - covid data
    cols = covid.columns.values.tolist()
    state_mappings = {} #maps mobility key to covid key
    with open('mainland.txt') as file:
        for line in file:
            code, name = line.split(',')
            state_mappings[code] = name[1:-1]
    fire = {} # May 16 - November 17, 2020
    fire_cov = {} 
    flood = {} # May 26 - December 5, 2021
    flood_cov = {} 
    collist = ['date', 'mobility_sum','apple_mobility', 'distance', 'cases', 'deaths']
    # separates data by state, disaster, and extracts necessary features
    for s in state_mappings:
        fire[s] = pd.DataFrame(columns=collist)
        flood[s] = pd.DataFrame(columns=collist)
    for s in state_mappings:
        state = mobility.loc[mobility['region'] == s] 
        statec = covid.loc[covid['state'] == state_mappings[s]] 
        fire_range = state[136:322]['date']
        fire_data = state[136:322][['date', 'retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline','apple_mobility']]
        #print(fire_data)
        flood_range = state[512:705]['date']
        flood_data = state[512:705][['date', 'retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline','apple_mobility']]
        for f in fire_range:
            k = f.split('/')
            yr = '20' + k[2]
            if len(k[0]) == 1:
                month = '0' + k[0]
            else:
                month = k[0]
            if len(k[1]) == 1:
                day = '0' + k[1]
            else:
                day = k[1]
            key = yr + '-' + month + "-" + day
            cov_data = statec.loc[statec['date'] == key]
            #fire_cov[s] = fire_cov[s].append(statec.loc[statec['date'] == key])
            
            #to add a row for a wildfire state dataset
            row = {}
            row['date'] = key   
            row['mobility_sum'] = (fire_data.loc[fire_data['date'] == f][['retail_and_recreation_percent_change_from_baseline']].values[0][0] + fire_data.loc[fire_data['date'] == f][['grocery_and_pharmacy_percent_change_from_baseline']].values[0][0] + fire_data.loc[fire_data['date'] == f][['parks_percent_change_from_baseline']].values[0][0] + fire_data.loc[fire_data['date'] == f][['transit_stations_percent_change_from_baseline']].values[0][0] + fire_data.loc[fire_data['date'] == f][['workplaces_percent_change_from_baseline']].values[0][0] + fire_data.loc[fire_data['date'] == f][['residential_percent_change_from_baseline']].values[0][0])
            row['apple_mobility'] = fire_data.loc[fire_data['date'] == f][['apple_mobility']].values[0][0]  
            # distance classification
            if s == 'CA':
                row['distance'] = 0
            elif s in ['NV', 'OR', 'AZ']:
                row['distance'] = 1
            else:
                row['distance'] = 2
            row['cases'] = cov_data.loc[cov_data['date'] == key][['cases']].values[0][0]
            row['deaths'] = cov_data.loc[cov_data['date'] == key][['deaths']].values[0][0]
            #print(row)
            fire[s] = fire[s].append(row, ignore_index=True)
        #print(fire[s])

        for f in flood_range:
            k = f.split('/')
            yr = '20' + k[2]
            if len(k[0]) == 1:
                month = '0' + k[0]
            else:
                month = k[0]
            if len(k[1]) == 1:
                day = '0' + k[1]
            else:
                day = k[1]
            key = yr + '-' + month + "-" + day
            cov_data = statec.loc[statec['date'] == key]
            #fire_cov[s] = fire_cov[s].append(statec.loc[statec['date'] == key])
            
            # to add a row to a hurricane/flood state dataset
            row = {}
            row['date'] = key   
            row['mobility_sum'] = (flood_data.loc[flood_data['date'] == f][['retail_and_recreation_percent_change_from_baseline']].values[0][0] + flood_data.loc[flood_data['date'] == f][['grocery_and_pharmacy_percent_change_from_baseline']].values[0][0] + flood_data.loc[flood_data['date'] == f][['parks_percent_change_from_baseline']].values[0][0] + flood_data.loc[flood_data['date'] == f][['transit_stations_percent_change_from_baseline']].values[0][0] + flood_data.loc[flood_data['date'] == f][['workplaces_percent_change_from_baseline']].values[0][0] + flood_data.loc[flood_data['date'] == f][['residential_percent_change_from_baseline']].values[0][0])
            row['apple_mobility'] = flood_data.loc[flood_data['date'] == f][['apple_mobility']].values[0][0]  
            # distance classification
            if s in ['LA', 'NJ', 'NY', 'PA', 'MS', 'AL', 'MD', 'CT']:
                row['distance'] = 0
            elif s in ['TX', 'AR', 'TN', 'GA', 'FL', 'OH', 'WV', 'DC', 'VA', 'DE', 'VT', 'MA', 'RI']:
                row['distance'] = 1
            else:
                row['distance'] = 2
            row['cases'] = cov_data.loc[cov_data['date'] == key][['cases']].values[0][0]
            row['deaths'] = cov_data.loc[cov_data['date'] == key][['deaths']].values[0][0]
            #print(row)
            flood[s] = flood[s].append(row, ignore_index=True)
        #print(flood[s])
    return fire, flood, state_mappings

if __name__ == "__main__":
    main()