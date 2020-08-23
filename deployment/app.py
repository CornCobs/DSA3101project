import streamlit as st
# run the command `streamlit run app.py` in terminal to start
import numpy as np
import pandas as pd
import time

import os
import glob
import multiprocessing

import plotly.express as px

st.title('DSA3101')

@st.cache
def load_data(nrows=None):
    # data = pd.read_csv("../data/data.csv", nrows=nrows)
    data = pd.read_csv("../data/DSA3101_Hackathon_Data.csv", nrows=nrows)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')
    return data

data = load_data()
panel_demo = pd.read_excel("../data/DSA3101_Hackathon_Panelists_Demographics.xlsx")

if st.checkbox('Show data'):
    st.subheader('First 1000 entries')
    st.dataframe(data.head(1000))

unique_dates = data.Date.unique()

st.subheader('RFM Modelling')
window = st.slider('Window (in Weeks)', 4, len(unique_dates)-1, 52)
score = st.slider('Score', 2, 6, 3)
weektofilter = st.slider('Week', window, len(unique_dates), len(unique_dates))  # min, max, default
refDateUpper = unique_dates[weektofilter-1]
refDateLower = unique_dates[weektofilter-window]
txt = f"Time intervals {refDateLower.astype(str)[:10]} to {refDateUpper.astype(str)[:10]}"

rfmModel = data[data['Date'] <= refDateUpper]
rfmModel = rfmModel[rfmModel['Date'] >= refDateLower].groupby(['Panel ID', 'Date']).agg(
    {'Spend': lambda x: x.sum()}).reset_index().groupby(['Panel ID']).agg(
    {'Date': lambda x: (refDateUpper-x.max()).days,    #Creating the RFM model dataframe
    'Panel ID': lambda x: len(x),
    'Spend': lambda x : x.mean()})

rfmModel.rename(columns={'Date' : 'Recency','Panel ID':'Frequency', 'Spend':'Monetary'}, inplace=True)
rfmModel = rfmModel.reset_index()
rfmModel.Recency /= 7

q = np.arange(start=1/score, stop=1, step=1/score)
if q[-1] == 1:
    q = q[:-1]
quantiles = rfmModel.quantile(q=q) # Get a dataframe of quantiles for each column
st.text("Quantile Cuts")
st.dataframe(quantiles)

quantiles = quantiles.to_dict() #Convert the dataframe into a nested dictionary of quantiles

def RScores(x,p,d):   #Functions to get the RFM scores. For R scores, the smaller the number the higher the rank (most recent)
    for i in range(len(q)):
        if x <= d[p][q[i]]:
            return len(q) - i + 1
    return 1
    
def FMScores(x,p,d):     #Function to get F and M scores. For F and M, the higher the rank, the more frequent and the more money
    for i in range(len(q)):
        if x <= d[p][q[i]]:
            return i + 1
    return len(q) + 1

#Apply a function along a column of the dataFrame.
rfmModel['R'] = rfmModel['Recency'].apply(RScores,args=('Recency',quantiles))
rfmModel['F'] = rfmModel['Frequency'].apply(FMScores,args=('Frequency',quantiles))
rfmModel['M'] = rfmModel['Monetary'].apply(FMScores,args=('Monetary',quantiles))
rfmModel['RFM'] = rfmModel.R.map(str) + rfmModel.F.map(str) + rfmModel.M.map(str)
rfmModel = rfmModel.sort_values(by='RFM').merge(panel_demo, left_on='Panel ID', right_on='ID')

fig = px.scatter_3d(rfmModel, x='Recency', y='Frequency', z='Monetary',
                    color='RFM', hover_name="Panel ID", 
                    hover_data=["BMI", "Income", "Ethnicity", "Lifestage", "Strata", "#HH", "location"])
fig.update_layout(title=txt, autosize=False,
                  width=800, height=800,
                  margin=dict(l=40, r=40, b=40, t=40))
fig.update_traces(marker=dict(size=4))
st.plotly_chart(fig)


#####################################################################################################################################

st.subheader('Change')
window = st.slider('Window (in Weeks)', 4, len(unique_dates)//2, 12)
score = st.slider('Score', 2, 5, 3)
week_final = st.slider('Week', 2*window, len(unique_dates), len(unique_dates))
field = st.selectbox(label="Filter by", options=("None", "Recency", "Frequency", "Monetary"))
refDateFinalUpper = unique_dates[week_final-1]
refDateFinalLower = unique_dates[week_final-window]
refDateInitUpper = unique_dates[week_final-window-1]
refDateInitLower = unique_dates[week_final-window-window]
txt = f"Comparing time periods {refDateInitLower.astype(str)[:10]} to {refDateInitUpper.astype(str)[:10]} with {refDateFinalLower.astype(str)[:10]} to {refDateFinalUpper.astype(str)[:10]}"

rfmModelFinal = data[data['Date'] <= refDateFinalUpper]
rfmModelFinal = rfmModelFinal[refDateFinalLower <= rfmModelFinal['Date']].groupby(['Panel ID', 'Date']).agg(
    {'Spend': lambda x: x.sum()}).reset_index().groupby('Panel ID').agg(
    {'Date': lambda x: (refDateFinalUpper-x.max()).days,    #Creating the RFM model dataframe
    'Panel ID': lambda x: len(x),
    'Spend': lambda x : x.mean()} )
rfmModelInit = data[data['Date'] <= refDateInitUpper]
rfmModelInit = rfmModelInit[refDateInitLower <= rfmModelInit['Date']].groupby(['Panel ID', 'Date']).agg(
    {'Spend': lambda x: x.sum()}).reset_index().groupby('Panel ID').agg(
    {'Date': lambda x: (refDateInitUpper-x.max()).days,    #Creating the RFM model dataframe
    'Panel ID': lambda x: len(x),
    'Spend': lambda x : x.mean()} )

rfmModelFinal.rename(columns={'Date' : 'Recency','Panel ID':'Frequency', 'Spend':'Monetary'}, inplace=True)
rfmModelFinal = rfmModelFinal.reset_index()
rfmModelFinal.Recency /= 7
rfmModelFinal.Recency = rfmModelFinal.Recency.map(int)

rfmModelInit.rename(columns={'Date' : 'Recency','Panel ID':'Frequency', 'Spend':'Monetary'}, inplace=True)
rfmModelInit = rfmModelInit.reset_index()
rfmModelInit.Recency /= 7
rfmModelInit.Recency = rfmModelInit.Recency.map(int)

q = np.arange(start=1/score, stop=1, step=1/score)
if q[-1] == 1:
    q = q[:-1]
quantilesFinal = rfmModelFinal.quantile(q=q) # Get a dataframe of quantiles for each column
quantilesFinal = quantilesFinal.to_dict() #Convert the dataframe into a nested dictionary of quantiles
quantilesInit = rfmModelInit.quantile(q=q) # Get a dataframe of quantiles for each column
quantilesInit = quantilesInit.to_dict() #Convert the dataframe into a nested dictionary of quantiles

#Apply a function along a column of the dataFrame.
rfmModelFinal['R'] = rfmModelFinal['Recency'].apply(RScores,args=('Recency',quantilesFinal))
rfmModelFinal['F'] = rfmModelFinal['Frequency'].apply(FMScores,args=('Frequency',quantilesFinal))
rfmModelFinal['M'] = rfmModelFinal['Monetary'].apply(FMScores,args=('Monetary',quantilesFinal))
rfmModelFinal['RFM'] = rfmModelFinal.R.map(str) + rfmModelFinal.F.map(str) + rfmModelFinal.M.map(str)
rfmModelFinal = rfmModelFinal.sort_values(by='RFM')
rfmModelFinal['State'] = 1
rfmModelInit['R'] = rfmModelInit['Recency'].apply(RScores,args=('Recency',quantilesInit))
rfmModelInit['F'] = rfmModelInit['Frequency'].apply(FMScores,args=('Frequency',quantilesInit))
rfmModelInit['M'] = rfmModelInit['Monetary'].apply(FMScores,args=('Monetary',quantilesInit))
rfmModelInit['RFM'] = rfmModelInit.R.map(str) + rfmModelInit.F.map(str) + rfmModelInit.M.map(str)
rfmModelInit = rfmModelInit.sort_values(by='RFM')
rfmModelInit['State'] = 0

rfmModelCombined = rfmModelFinal.append(rfmModelInit)

if field == "None":
    rfmModelCombined_df = rfmModelCombined.pivot(index='Panel ID',columns='State').dropna()
    rfmModelCombined_df = rfmModelCombined_df[rfmModelCombined_df['RFM'][0] != rfmModelCombined_df['RFM'][1]]
    rfmModelCombined_df['ChangeR'] = np.sign(rfmModelCombined_df["R"][1] - rfmModelCombined_df["R"][0])
    rfmModelCombined_df['ChangeF'] = np.sign(rfmModelCombined_df["F"][1] - rfmModelCombined_df["F"][0])
    rfmModelCombined_df['ChangeM'] = np.sign(rfmModelCombined_df["M"][1] - rfmModelCombined_df["M"][0])
    rfmModelCombined_df['Change'] = rfmModelCombined_df['ChangeR'].map(str) + ", " + \
        rfmModelCombined_df['ChangeF'].map(str) + ", " + rfmModelCombined_df['ChangeM'].map(str)
    rfmModelCombined_df = rfmModelCombined_df[['Change']].droplevel('State', axis=1).reset_index()
    rfmModelCombined_df2 = rfmModelCombined[rfmModelCombined['Panel ID'].isin(rfmModelCombined_df['Panel ID'])].merge(rfmModelCombined_df, on='Panel ID')

else:
    prefix = field[0]
    rfmModelCombined_df = rfmModelCombined.pivot(index='Panel ID',columns='State').dropna()
    rfmModelCombined_df[f'Change{prefix}'] = np.sign(rfmModelCombined_df[prefix][1] - rfmModelCombined_df[prefix][0])
    rfmModelCombined_df = rfmModelCombined_df[rfmModelCombined_df[f'Change{prefix}'] != 0]
    rfmModelCombined_df = rfmModelCombined_df[[f'Change{prefix}']].droplevel('State', axis=1).reset_index()
    rfmModelCombined_df2 = rfmModelCombined[rfmModelCombined['Panel ID'].isin(rfmModelCombined_df['Panel ID'])].merge(rfmModelCombined_df, on='Panel ID')

change_dict= {'Recency': 'ChangeR', 'Frequency': 'ChangeF', 'Monetary': 'ChangeM', 'None': 'Change'}
rfmModelCombined_df2['State'] = rfmModelCombined_df2['State'].map({0: 'Before', 1:'After'})
rfmModelCombined_df2 = rfmModelCombined_df2.merge(panel_demo, left_on='Panel ID', right_on='ID')

fig = px.line_3d(rfmModelCombined_df2, x='Recency', y='Frequency', z='Monetary', color=change_dict[field], line_group='Panel ID',
                hover_name="Panel ID", 
                hover_data=["RFM", "State", "BMI", "Income", "Ethnicity", "Lifestage", "Strata", "#HH", "location"], 
                labels={"ChangeR": "Recency", "ChangeF": "Frequency", "ChangeM": "Monetary"})
                
fig.update_layout(title=txt, autosize=False,
                  width=800, height=800,
                  margin=dict(l=40, r=40, b=40, t=40))
st.plotly_chart(fig)

#####################################################################################################################################

st.subheader("Panel's Behaviour over Time")
window = st.slider('Window (in Weeks)', 4, len(unique_dates)-1, 104)
score = st.slider('Score', 2, 5, 5)
cutoff = st.slider('Cutoff week', window+1, len(unique_dates), len(unique_dates))
panel_id = st.selectbox(label="Panel ID", options=sorted(data['Panel ID'].unique()))
panel_data = pd.DataFrame()

for weektofilter in range(window, cutoff):
    refDateUpper = unique_dates[weektofilter-1]
    refDateLower = unique_dates[weektofilter-window]

    rfmModel = data[data['Date'] <= refDateUpper]
    rfmModel = rfmModel[rfmModel['Date'] >= refDateLower].groupby(['Panel ID', 'Date']).agg(
    {'Spend': lambda x: x.sum()}).reset_index().groupby('Panel ID').agg(
        {'Date': lambda x: (refDateUpper-x.max()).days,    #Creating the RFM model dataframe
        'Panel ID': lambda x: len(x),
        'Spend': lambda x : x.mean()})

    rfmModel.rename(columns={'Date' : 'Recency','Panel ID':'Frequency', 'Spend':'Monetary'}, inplace=True)
    rfmModel = rfmModel.reset_index()
    rfmModel.Recency /= 7

    q = np.arange(start=1/score, stop=1, step=1/score)
    if q[-1] == 1:
        q = q[:-1]
    quantiles = rfmModel.quantile(q=q) # Get a dataframe of quantiles for each column
    quantiles = quantiles.to_dict() #Convert the dataframe into a nested dictionary of quantiles

    #Apply a function along a column of the dataFrame.
    rfmModel['R'] = rfmModel['Recency'].apply(RScores,args=('Recency',quantiles))
    rfmModel['F'] = rfmModel['Frequency'].apply(FMScores,args=('Frequency',quantiles))
    rfmModel['M'] = rfmModel['Monetary'].apply(FMScores,args=('Monetary',quantiles))
    rfmModel['RFM'] = rfmModel.R.map(str) + rfmModel.F.map(str) + rfmModel.M.map(str)
    # rfmModel = rfmModel.sort_values(by='RFM')
    rfmModel['Date'] = refDateUpper
    panel_data = panel_data.append(rfmModel[['Date', 'Recency', 'Frequency', 'Monetary', 'RFM']][rfmModel['Panel ID'] == panel_id])

first = str(panel_data.Date.tolist()[0])[:-9]
after = panel_data.RFM.tolist()[-1]
before = panel_data.RFM.tolist()[-2]
last = str(panel_data.Date.tolist()[-1])[:-9]
if before == after:
    txt = f"{panel_id}'s RFM score has no change in the last week"
else:
    txt = f"{panel_id}'s RFM changes from {before} to {after} in the last week"

st.dataframe(panel_demo[panel_demo['ID'] == panel_id])

st.text(f"Time period: {first} to {last}")

fig = px.line_3d(panel_data, x='Recency', y='Frequency', z='Monetary',
                 hover_name="Date", hover_data=["RFM"])
fig.update_layout(title=txt, autosize=False,
                  width=800, height=800,
                  margin=dict(l=40, r=40, b=40, t=40))
fig.update_traces(marker=dict(size=4))
st.plotly_chart(fig)









