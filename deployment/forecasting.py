### This is just a test run of me using a streamlit application

import streamlit as st
# run the command `streamlit run app.py` in terminal to start
import numpy as np
import pandas as pd
import time

import os
import glob
import multiprocessing

import plotly.express as px

from sklearn.neighbors import BallTree

import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns

from fastai.imports import *
from fbprophet import Prophet

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

import statsmodels.api as sm
# Initialize plotly
init_notebook_mode(connected=True)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

np.random.seed(0)

st.title('Demand Forecasting App')

@st.cache

## Processing data

data= pd.read_csv("../data/data.csv")
new= pd.read_excel("../data/DSA3101_Hackathon_Panelists_Demographics.xlsx")
data["ID"] = data["Panel ID"]
data.drop("Panel ID", axis=1)
df = pd.merge(data,new,how = 'inner', on = ['ID'])
df = df.drop(["Strata","Product","Pack Size","Volume","Spend","Ratio","ID","BMI","Income","Ethnicity","Lifestage","#HH","Panel ID"],axis = 1)
df = df.groupby(['Category', 'location', 'Date']).size().reset_index(name='sales')

forecast = pd.read_csv("../data/forecast.csv")
#####################################################################################################################################






def show_forecast(cmp_df, num_predictions, num_values, title):
    """Visualize the forecast."""
    
    def create_go(name, column, num, **kwargs):
        points = cmp_df.tail(num)
        args = dict(name=name, x=points.index, y=points[column], mode='lines')
        args.update(kwargs)
        return go.Scatter(**args)
    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="aqua"))
    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,
                            line=dict(width=0),
                            marker=dict(color="aqua"),
                            fillcolor='rgba(68, 68, 68, 0.3)', 
                            fill='tonexty')
    forecast = create_go('Forecast', 'yhat', num_predictions,
                         line=dict(color='rgb(31, 119, 180)'))
    actual = create_go('Actual', 'y', num_values,
                       marker=dict(color="red"))
    
    # In this case the order of the series is important because of the filling
    data = [lower_bound, upper_bound, forecast, actual]

    layout = go.Layout(yaxis=dict(title='sales'), title=title, showlegend = False)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)



st.plotly_chart(show_forecast(cmp_df, prediction_size, 100, 'Sales on Store $1$ for Item $1$'))