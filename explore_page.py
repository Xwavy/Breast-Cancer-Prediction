import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image

@st.cache_data
def load_data():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df = df.dropna()
    df.replace('?', -99999, inplace=True)
    df.astype(float).values.tolist()
    df.drop(labels='id', axis=1, inplace=True)
    return df


def show_explore_page():
    st.title('Breast Cancer Prediction Model By Philip Olufunmilayo')
    img2 = Image.open('breast-cancer-symptoms.jpg')
    st.image(img2)
    st.header('Information on the Dataset')
    st.write("""
    The breast cancer databases used to build this model, was obtained from the University of Wisconsin
   Hospitals, Madison from Dr. William H. Wolberg. The citations are as follows:

   1. O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear 
      programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

   2. William H. Wolberg and O.L. Mangasarian: "Multisurface method of 
      pattern separation for medical diagnosis applied to breast cytology", 
      Proceedings of the National Academy of Sciences, U.S.A., Volume 87, 
      December 1990, pp 9193-9196.

   3. O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition 
      via linear programming: Theory and application to medical diagnosis", 
      in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying
      Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

   4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming 
      discrimination of two linearly inseparable sets", Optimization Methods
      and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).
    
    """)
    st.write("Here are the performance metrics of the model:")
    st.write("Accuracy score: 0.9785714285714285")
    st.write("Precision score: 0.9797979797979798")
    st.write("Recall score: 0.9897959183673469")
    st.write("""Confusion matrix: [[97  1]
                                    [ 2 40]]""")
    st.write("F1 score: 0.9847715736040609")
    st.subheader('This shows the correlation of the features in the dataset')
    img = Image.open('corr breast.png')
    st.image(img, caption='correlation of the features')
    st.subheader('Disribution of the classes of tumors in the dataset')
    img1 = Image.open('breast dist.png')
    st.image(img1, caption='Disribution of the classes of tumors in the dataset')


