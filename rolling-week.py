# Libraries used
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.stattools import acf, pacf
import numpy as np

def CleanData(DF):
    '''
        Modify Date Value to convert to datetime accepted object + Convert text data into numerical values

    :param DF: Raw data in dataframe
    :return: Numerical dataframe besides Dates
    '''
    originalDate = []
    YearMonthDate = []
    vol = []

    for idx in range(len(DF)):
        # Read in Date and Volume value as string
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

def ObservationsByTimeWindow(commodity, DataDF):
    """

    :param commodity: Silver or Gold
    :param DataDF: Clean Dataframe from chosen commodity
    :return: Plot resampled rolling mean by different time window for chosen commodity on Price and Volume
    """
    # Resample Original Data with different time window - daily/weekly/monthly
    daily_mean = DataDF[data_columns].resample('B').mean()
    weekly_mean = DataDF[data_columns].resample('W').mean()
    monthStart_mean = DataDF[data_columns].resample('M').mean()

    # Calculate Rolling Mean + Centered 7-day and Yearly
    sevenDayRollingMean = DataDF[data_columns].rolling(window=7, center=True).mean()
    yearRollingMean = DataDF[data_columns].rolling(window=365, center=True).mean()

    # Range of timeframe for Rolling Resampling Mean - Price and Volume
    startDate, endDate = '2018-01', '2019-01'
    startYear, endYear = '2015-01', '2019-01'

    # Plot Re-sampled Rolling Mean Graph for Observation on Price
    mpl.style.use('seaborn')  # Use Gray Background on Figure
    PriceFig, ax_Price = plt.subplots()
    ax_Price.plot(daily_mean.loc[startDate:endDate, 'Price'],
                  marker='.', linestyle='-', linewidth=0.5, label='Daily', color='steelblue')
    ax_Price.plot(weekly_mean.loc[startDate:endDate, 'Price'],
                  marker='o', markersize=5, linestyle='-', label='Weekly Mean Resample', color='forestgreen')
    ax_Price.plot(sevenDayRollingMean.loc[endDate:startDate, 'Price'],
                  marker='*', markersize=6, linestyle='-', linewidth=1, label='7-Day Rolling Mean', color='tomato')
    # Set Labels / Legends
    plt.title(commodity + ' Price Rolled-up by Different Time Window', fontsize=12)
    ax_Price.set_ylabel('Price ($)')
    ax_Price.legend()
    # Set Output Figure Size and Name
    PriceFig.set_size_inches(18.5, 10)
    PriceFig.savefig(commodity + " Price Time Series.png")

    # Plot Re-sampled Rolling Mean Graph for Observation on Volume by Year
    mpl.style.use('seaborn')
    VolFig, ax_Volume = plt.subplots()
    ax_Volume.plot(daily_mean.loc[startYear:endYear, 'Vol.'],
                   marker='.', linestyle='None', label='Daily', color='steelblue')
    ax_Volume.plot(sevenDayRollingMean.loc[endYear:startYear, 'Vol.'],
                   linestyle='-', label='7-Day Rolling Mean', color='forestgreen')
    ax_Volume.plot(yearRollingMean.loc[endYear:startYear, 'Vol.'],
                   linestyle='-', linewidth=1, label='Yearly Rolling Mean', color='tomato')
    # Set Output Figure Name
    VolFig.savefig(commodity + " Volume Time Series.png")


def ObservationACF_PACF(commodity, DataDF):
    """
    Plot ACF and PACF graphs for Silver or Gold Price

    :param commodity: Silver or Gold
    :param DataDF: Clean Dataframe from chosen commodity
    :return: Generate ACF and PACF figures for chosen commodity
    """
    # ACF + PACF Evaluation
    priceData = DataDF['Price'].reset_index(drop=True)
    # Split Dat into Train and Test
    percentage_training = 0.8
    numLags = 15

    split_point = round(len(priceData) * percentage_training)
    train, test = priceData[0:split_point], priceData[split_point:]
    # plt.plot(train)
    train = np.log(train)
    train_diff = train.diff(periods=1).values[1:]
    plt.plot(train_diff)

    # Calculate ACF and PACF
    lag_acf = acf(train_diff, nlags=numLags)
    lag_pacf = pacf(train_diff, nlags=numLags, method='ols')

    # Plot ACF
    mpl.style.use('seaborn')
    ACF_PACF_fig = plt.figure(figsize=(15, 5))
    ax1 = ACF_PACF_fig.add_subplot(121)
    ax1.title.set_text(commodity + ' Autocorrelation')
    plt.subplot(121)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(train)), linestyle='--', color='tomato')
    plt.axhline(y=1.96 / np.sqrt(len(train)), linestyle='--', color='tomato')
    plt.xlabel("Lag")
    plt.ylabel("ACF")

    # Plot PACF
    ax2 = ACF_PACF_fig.add_subplot(122)
    ax2.title.set_text(commodity + ' Partial Autocorrelation')
    plt.subplot(122)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(train)), linestyle='--', color='tomato')
    plt.axhline(y=1.96 / np.sqrt(len(train)), linestyle='--', color='tomato')
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.tight_layout()

    ACF_PACF_fig.savefig(commodity + " ACF & PACF.png")



# --------------------------------------------- MAIN ---------------------------------------------
# Run the Observation for both Commodity
commodity = ["Silver", "Gold"]

for idx in range(len(commodity)):
    # Read data from csv to dataframe
    Data = pd.read_csv("Data\\" + commodity[idx] + " Futures Historical Data.csv")
    # Clean up data - convert text data into numbers
    DataDF = CleanData(Data)
    # Set Date as index
    DataDF.set_index(pd.DatetimeIndex(DataDF['Date']), inplace=True)
    DataDF = DataDF.drop(['Date', 'Change %'], axis=1)
    data_columns = list(DataDF.columns)

    # Observations by different time window
    ObservationsByTimeWindow(commodity[idx], DataDF)

    # Generate ACF + PACF for each commodity
    ObservationACF_PACF(commodity[idx], DataDF)



'''
# SARIMA Model Built
SARIMA_model = SARIMAX(train, order=(0, 1, 1), enforce_stationarity=False, enforce_invertibility=False, trend='c')
# Extract Fitted Model
SARIMA_model_fit = SARIMA_model.fit(disp=False)
print(SARIMA_model_fit.summary())
'''