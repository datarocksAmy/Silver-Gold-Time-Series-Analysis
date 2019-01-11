# Library used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sb
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def CleanData(DF):
    '''
        Modify Date Value to convert to datetime accepted object
    :param DF: Raw data in dataframe
    :return:
    '''
    originalDate = []
    YearMonthDate = []
    vol = []

    for idx in range(len(DF)):
        # Read in Date and Volume value as string
        #dateString = str(DF['Date'][idx]).replace(",", "").split(" ")
        dateString = datetime.strptime(DF['Date'][idx],"%m/%d/%Y" ).strftime("%Y-%m-%d")
        YMdateString = datetime.strptime(DF['Date'][idx], "%m/%d/%Y").strftime("%Y-%m")
        volString = str(DF['Vol.'][idx])

        # Transform "K" in Vol with trailing 0s to convert text into number
        if "K" in volString:
            numVol = float(volString.replace("K", "")) * 1000
            vol.append(numVol)
        else:
            vol.append(0)
        # Append to the corresponding lists
        originalDate.append(dateString)
        YearMonthDate.append(YMdateString)

    # Modify/Update original Dataframe
    DF['Date'] = originalDate
    DF['YearMonth'] = YearMonthDate
    DF['Vol.'] = vol

    return DF

rolling_increment = 7


GoldData = pd.read_csv("Data\Silver Futures Historical Data.csv")
GoldDataDF = CleanData(GoldData)


df = GoldDataDF
df.YearMonth = pd.to_datetime(df.YearMonth)

# change to index
df.set_index('YearMonth', inplace=True)

Price = df[['Price']]
#print(GoldDataDF['Price'])
Price.rolling(rolling_increment).mean().plot(figsize=(20,10), linewidth=2, fontsize=10)
plt.xlabel('Year', fontsize=30)
#plt.show()

Price.diff().plot(figsize=(20,10), linewidth=3, fontsize=5)
plt.xlabel('Year', fontsize=5)
#plt.show()


GoldDataDF['Natural Log'] = GoldDataDF['Price'].apply(lambda x: np.log(x))

GoldDataDF['Original Variance'] = Price.rolling(rolling_increment).var()
GoldDataDF['Log Variance'] = GoldDataDF['Natural Log'].rolling(rolling_increment).var()

fig, ax = plt.subplots(2, 1, figsize=(13, 12))
GoldDataDF['Original Variance'].plot(ax=ax[0], title='Original Variance')
GoldDataDF['Log Variance'].plot(ax=ax[1], title='Log Variance')
#fig.tight_layout()
fig.savefig("test.png")

GoldDataDF['Logged First Difference'] = GoldDataDF['Natural Log'] - GoldDataDF['Natural Log'].shift()
GoldDataDF['Lag 1'] = GoldDataDF['Logged First Difference'].shift()
GoldDataDF['Lag 2'] = GoldDataDF['Logged First Difference'].shift(2)
GoldDataDF['Lag 5'] = GoldDataDF['Logged First Difference'].shift(5)
GoldDataDF['Lag 30'] = GoldDataDF['Logged First Difference'].shift(30)
sb.jointplot('Logged First Difference', 'Lag 1', GoldDataDF, kind='reg', size=13)


lag_correlations = acf(GoldDataDF['Logged First Difference'].iloc[1:])
lag_partial_correlations = pacf(GoldDataDF['Logged First Difference'].iloc[1:])


fig, ax = plt.subplots(figsize=(16,12))
ax.plot(lag_correlations, marker='o', linestyle='--')

decomposition = seasonal_decompose(GoldDataDF['Natural Log'], model='additive', freq=30)
figSeasonalDecompose = plt.figure()
figSeasonalDecompose = decomposition.plot()




plt.show()
