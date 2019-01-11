# Library used
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt, log
import matplotlib.pylab as plt
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
import warnings
import datetime
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import bootstrap_point632_score
import matplotlib as mpl
import pickle
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric

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



warnings.filterwarnings('ignore')
GoldData = pd.read_csv("Data\Gold Futures Historical Data.csv")
GoldDataDF = CleanData(GoldData)
df = GoldDataDF[['Date', 'Price']]
#print(df.head())
df.columns = ["ds", "y"]
R2List = []


#for i in range(5):
train_resid = []
test_resid = []

# Split Training and Testing Data
#train, test = train_test_split(df, test_size=0.2)
percentage_training = 0.7
split_point = round(len(df) * percentage_training)
train, test = df[0:split_point], df[split_point:]

# Prophet Model
m = Prophet(interval_width=0.95)

# Fit the training dataset
#m.fit(train_df2)
m.fit(train)

# Predict
train_future = m.make_future_dataframe(periods=len(test))
train_forecast = m.predict(train_future)

metric_df = train_forecast.set_index('ds')[['yhat']].join(train.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

# Calculate RMSE and MSE
Train_RMSE = sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
Train_MSE = mean_squared_error(metric_df.y, metric_df.yhat)

# Calculating AIC for training dataset
for num in range(len(metric_df.y)):
    train_resid.append(metric_df.y[num] - metric_df.yhat[num])

train_SSE = sum([num**2 for num in train_resid])
train_numVar = len(test)
train_AIC = 2*train_numVar - 2*log(train_SSE)

R2List.append(r2_score(metric_df.y, metric_df.yhat))

print("--------------------------------")
print("<<<<< PROPHET >>>>>")
print("RMSE: %.2f" % Train_RMSE)
#print("MSE: %.2f" % Train_MSE)
print("R2 score : %.2f" % r2_score(metric_df.y, metric_df.yhat))
print("AIC: %.2f" % train_AIC)

actualDF = metric_df[['ds', 'y']]
actualDF.set_index(actualDF['ds'], inplace=True)
actualDF = actualDF.drop(['ds'], axis=1)

predictDF = metric_df[['ds', 'yhat']]
predictDF.set_index(predictDF['ds'], inplace=True)
predictDF = predictDF.drop(['ds'], axis=1)

mpl.style.use('seaborn')
fig, ax = plt.subplots()

ax.plot(actualDF,
        marker='.', linestyle='--', markersize=2, linewidth=0.5, label='Actual', color= 'steelblue')
ax.plot(predictDF,
        marker='o', markersize=1, linestyle='-', label='Predicted', color='tomato')
# Set Labels / Legends
fig.suptitle('Prophet Model Predict v.s. Actual Price', fontsize=12)
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.legend()

fig.savefig("Prophet Predict v.s. Actual.png")

# Pickle the Trained/Tested Prophet Model
#with open("Silver_Forecaster.pickle", "wb") as AF:
    #pickle.dump(m, AF)


df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_p = performance_metrics(df_cv)
print(df_p.head())

# Can only run on Win64 bit system
#MAPEfig = plot_cross_validation_metric(df_cv, metric='mape')
#MAPEfig.savefig("MAPE.png")
