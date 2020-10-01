#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:56:06 2020

@author: chengling
"""
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

np.random.seed(0)

st.title('LETS GO SHOPEE PEE PEE PEE!')

@st.cache
def load_data(nrows=None):
    data = pd.read_csv("../data/data.csv", nrows=nrows, parse_dates=[1])
    return data

data = load_data()
panel_demo = pd.read_excel("../data/DSA3101_Hackathon_Panelists_Demographics.xlsx")
U = pd.read_csv("../models/U.csv")
V = pd.read_csv("../models/V.csv")
panels = U['Panel ID']
products = V['Product']
d = U.shape[1]

# UV = pd.DataFrame(UV, index=panels, columns=products)

#####################################################################################################################################

panel_id = st.selectbox(label="Panel ID", options=sorted(data['Panel ID'].unique()), index=1)

UV = (U[U['Panel ID'] == panel_id].values[:, 1:] @ V.values[:, 1:].T).ravel()

num_entries = int(st.text_input("Max no. of entries: ", 10))
st.subheader(f"[Historical Purchases] Your last {num_entries} purchases")
history = data[data['Panel ID'] == panel_id] \
    .sort_values(by=['Week', 'Spend'], ascending=[False, False])[['Product', 'Date', 'Spend']] \
    .drop_duplicates() \
    .head(num_entries) \
    .reset_index(drop=True)
st.table(history.style.format({'Spend': '{:.2f}'}))
st.text(f"Would you like to buy them again? :)")

#####################################################################################################################################

st.subheader(f"[Current Discounts/Promotions] HOT/FLASH DEALS")
st.text(f"""Sort by stock inventory availability from most to least 
and perishable goods to non perishable goods""")

#####################################################################################################################################

st.subheader(f"[Current Demand] TRENDING ITEMS")
window = st.slider('Window (in Weeks)', 1, 156, 1)
k = int(st.text_input("Number of products: ", 5))
st.text(f"Top {k} sales in the past {window} week(s)")
trending = data[data.Week >= 156-window][['Product', 'Category']].groupby(['Product']).agg(['count'])
trending.columns = ['Count']
st.table(trending  \
    .sort_values(by=['Count'], ascending=[False]) \
    .head(k).reset_index())

#####################################################################################################################################

st.subheader(f"[Historical Views] Recently viewed")
st.table(data.Product.sample(5).reset_index(drop=True))
st.text(f"You have recently viewed these products. Would you like to purchase them :)")

#####################################################################################################################################

st.subheader(f"[User Similarity] See what products are your like-minded peers buying")
num_panels = int(st.text_input("Number of similar panels: ", 5))
tree = BallTree(U[[f"X{_}" for _ in range(1, d)]], leaf_size=30)
_, ind = tree.query(U[U['Panel ID']==panel_id][[f"X{_}" for _ in range(1, d)]], k=num_panels+1)

similar_panels = [panels[_] for _ in ind[0] if panels[_] != panel_id]
        
st.table(data[data['Panel ID'].isin(similar_panels)] [['Panel ID', 'Product', 'Date', 'Spend']] \
    .drop_duplicates() \
    .groupby('Panel ID') \
    .head(1) \
    .sort_values(by=['Date', 'Spend'], ascending=[False, False]) \
    .reset_index(drop=True) \
    .style.format({'Spend': '{:.2f}'}))

#####################################################################################################################################

st.subheader(f"[Item Similarity] See what products are similar to your past purchases")
num_items = int(st.text_input("Number of similar products: ", 5))
tree = BallTree(U[[f"X{_}" for _ in range(1, d)]], leaf_size=30)
product_list = list(history.Product)
    
similar_items = []
for prod in product_list:
    _, ind = tree.query(V[V['Product']==prod][[f"X{_}" for _ in range(1, d)]], k=max(num_items//num_entries+1, 1))
    similar_items.extend([products[_] for _ in ind[0] if products[_] not in product_list])
        
similar_items_df = data[data['Product'].isin(similar_items)] [['Product', 'Spend', 'Volume', 'Pack Size', 'Category']] \
    .drop_duplicates() \
    .groupby('Product') \
    .head(1) \
    .sort_values(by=['Spend'], ascending=[False]) \
    .reset_index(drop=True) 
    
if num_items < len(similar_items_df):
    similar_items_df = similar_items_df[:num_items]    

similar_items_df['Spend'] = similar_items_df['Spend']/similar_items_df['Pack Size']
similar_items_df['Volume'] = similar_items_df['Volume']/similar_items_df['Pack Size']

st.table(similar_items_df[['Product', 'Spend', 'Volume', 'Category']] \
         .style.format({'Spend': '{:.2f}', 'Volume': '{:.2f}'}))
    

#####################################################################################################################################

st.subheader(f"You may also like")
num_prod = int(st.text_input("No. of products: ", 5))
st.table(products[np.argsort(UV)][-num_prod:][::-1].drop_duplicates() \
        .head(num_prod) \
        .reset_index(drop=True))
