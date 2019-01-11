import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import numpy as np

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
    #DF['YearMonth'] = YearMonthDate
    DF['Vol.'] = vol

    return DF


GoldData = pd.read_csv("Data\Gold Futures Historical Data.csv")
GoldDataDF = CleanData(GoldData)
GoldDataDF.set_index(pd.DatetimeIndex(GoldDataDF['Date']), inplace=True)
GoldDataDF = GoldDataDF.drop(['Date', 'Change %'], axis=1)
daily_df = GoldDataDF
data_columns = list(GoldDataDF.columns)


# Resample original data to weekly frequency, aggregate with mean
daily_mean = GoldDataDF[data_columns].resample('B').mean()
weekly_mean = GoldDataDF[data_columns].resample('W').mean()
monthStart_mean = GoldDataDF[data_columns].resample('M').mean()

# Cetnered 7-day rolling mean
sevenDayRollingMean = GoldDataDF[data_columns].rolling(window=7, center=True).mean()
yearRollingMean = GoldDataDF[data_columns].rolling(window=365, center=True).mean()

# Start and end of the date range to extract
start, end = '2018-01', '2019-01'
# Plot daily and weekly resampled time series together
mpl.style.use('seaborn')
PriceFig, Pax = plt.subplots()
Pax.plot(daily_mean.loc[start:end, 'Price'],
        marker='.', linestyle='-', linewidth=0.5, label='Daily', color= 'steelblue')
Pax.plot(weekly_mean.loc[start:end, 'Price'],
        marker='o', markersize=5, linestyle='-', label='Weekly Mean Resample', color='forestgreen')
Pax.plot(sevenDayRollingMean.loc[end:start, 'Price'],
        marker='*', markersize=6, linestyle='-', linewidth=1, label='7-Day Rolling Mean', color='tomato')
# Set Labels / Legends
Pax.set_ylabel('Price ($)')
Pax.legend()

# Set Output Figure Size and Name
PriceFig.set_size_inches(18.5, 10)
PriceFig.savefig("Price Time Series.png")

startYear, endYear = '2015-01', '2019-01'
mpl.style.use('seaborn')
VolFig, Vax = plt.subplots()
Vax.plot(daily_mean.loc[startYear:endYear, 'Vol.'],
        marker='.', linestyle='None', label='Daily', color= 'steelblue')
Vax.plot(sevenDayRollingMean.loc[endYear:startYear, 'Vol.'],
        linestyle='-', label='Weekly Mean Resample', color='forestgreen')
Vax.plot(yearRollingMean.loc[endYear:startYear, 'Vol.'],
        linestyle='-', linewidth=1, label='7-Day Rolling Mean', color='tomato')

VolFig.savefig("Volume Time Series.png")
#plt.show()

#print(GoldDataDF.head())
series = GoldDataDF['Price'].reset_index(drop=True)

percentage_training = 0.8
split_point = round(len(series) * percentage_training)
train, test = series[0:split_point], series[split_point:]
#plt.plot(train)
train = np.log(train)
train_diff = train.diff(periods=1).values[1:]
plt.plot(train_diff)

lag_acf = acf(train_diff, nlags=15)
lag_pacf = pacf(train_diff, nlags=15, method='ols')

# Plot ACF
mpl.style.use('seaborn')
ACF_PACF_fig = plt.figure(figsize=(15, 5))
ax1 = ACF_PACF_fig.add_subplot(121)
ax1.title.set_text('Autocorrelation')
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train)), linestyle='--', color='tomato')
plt.axhline(y=1.96/np.sqrt(len(train)), linestyle='--', color='tomato')
plt.xlabel("Lag")
plt.ylabel("ACF")

# Plot PACF
ax2 = ACF_PACF_fig.add_subplot(122)
ax2.title.set_text('Partial Autocorrelation')
plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=-1.96/np.sqrt(len(train)), linestyle='--', color='tomato')
plt.axhline(y=1.96/np.sqrt(len(train)), linestyle='--', color='tomato')
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.tight_layout()

ACF_PACF_fig.savefig("ACF & PACF.png")
#plt.show()


# SARIMA Model Built
SARIMA_model = SARIMAX(train, order=(0, 1, 1), enforce_stationarity=False, enforce_invertibility=False, trend='c')
# Extract Fitted Model
SARIMA_model_fit = SARIMA_model.fit(disp=False)
print(SARIMA_model_fit.summary())