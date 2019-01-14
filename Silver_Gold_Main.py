from Prophet_Model import ProphetModel
from TS_Analysis import CleanData, TS_Main
from Prophet_Model import ProphetModel
import pandas as pd


# Run the Observation and Prophet Model for both Commodity
commodity = ["Silver", "Gold"]

for idx in range(len(commodity)):
    # Read data from csv to dataframe
    Data = pd.read_csv("Data\\" + commodity[idx] + " Futures Historical Data.csv")
    # Clean up data - convert text data into numbers
    DataDF = CleanData(Data)

    # Time Series Analysis + Observation
    TS_Main(commodity[idx], DataDF)

    # Build Prophet Model
    ProphetModel(commodity[idx], DataDF)