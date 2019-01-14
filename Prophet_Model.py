# Library used
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt, log
import matplotlib.pylab as plt
import warnings
from datetime import datetime
import matplotlib as mpl
import pickle



def ProphetModel(commodity, cleanDF):
    """
        Predict commodity future prices
    :param commodity: Silver or Gold
    """

    # Don't show warning in console
    warnings.filterwarnings('ignore')
    # Variable Initialization
    train_resid = []
    test_resid = []
    R2List = []
    # Obtain Date and Price data from DF
    df = cleanDF[['Date', 'Price']]
    # Change column names to fit the rules in Prophet Model
    df.columns = ["ds", "y"]

    # Split Training and Testing Data
    percentage_training = 0.7
    split_point = round(len(df) * percentage_training)
    train, test = df[0:split_point], df[split_point:]

    # Prophet Model
    model = Prophet(interval_width=0.95)
    # Fit the training data to model
    model.fit(train)
    # Predict
    train_future = model.make_future_dataframe(periods=len(test))
    train_forecast = model.predict(train_future)
    # Reset dataframe and put predict v.s. actual values side by side
    metric_df = train_forecast.set_index('ds')[['yhat']].join(train.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)

    # Calculate RMSE and MSE
    RMSE = sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    MSE = mean_squared_error(metric_df.y, metric_df.yhat)

    # Calculating AIC for training dataset
    for num in range(len(metric_df.y)):
        train_resid.append(metric_df.y[num] - metric_df.yhat[num])
    train_SSE = sum([num**2 for num in train_resid])
    train_numVar = len(test)
    AIC = 2*train_numVar - 2*log(train_SSE)
    R2List.append(r2_score(metric_df.y, metric_df.yhat))

    # Output RMSE, R2 Score and AIC Value for Prophet Model
    print("--------------------------------")
    print("<<<<< {} PROPHET >>>>>".format(commodity))
    print("RMSE: %.2f" % RMSE)
    #print("MSE: %.2f" % Train_MSE)
    print("R2 score : %.2f" % r2_score(metric_df.y, metric_df.yhat))
    print("AIC: %.2f" % AIC)

    # For graph purpose : Extract actual and predict to two separate dataframes
    actualDF = metric_df[['ds', 'y']]
    actualDF.set_index(actualDF['ds'], inplace=True)
    actualDF = actualDF.drop(['ds'], axis=1)
    predictDF = metric_df[['ds', 'yhat']]
    predictDF.set_index(predictDF['ds'], inplace=True)
    predictDF = predictDF.drop(['ds'], axis=1)

    # Plot Actual v.s. Predict
    # Set graph background as grey
    mpl.style.use('seaborn')
    preduct_acutal_fig, ax = plt.subplots()
    ax.plot(actualDF,
            marker='.', linestyle='--', markersize=2, linewidth=0.5, label='Actual', color= 'steelblue')
    ax.plot(predictDF,
            marker='o', markersize=1, linestyle='-', label='Predicted', color='tomato')
    # Set Labels / Legends
    preduct_acutal_fig.suptitle('{} Prophet Model Predict v.s. Actual Price'.format(commodity), fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price ($)')
    ax.legend()
    # Save the figure
    preduct_acutal_fig.savefig(commodity + " Prophet Predict v.s. Actual.png")

    # Cross Validate Model
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)
    print(df_p)

    # Pickle the Trained/Tested Prophet Model for further use
    prophetModelPickleName = "{}_Forecaster.pickle".format(commodity)
    #with open(prophetModelPickleName, "wb") as AF:
        #pickle.dump(m, AF)

