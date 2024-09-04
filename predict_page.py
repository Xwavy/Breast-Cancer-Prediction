import streamlit as st
import pickle
import pandas as pd
import numpy as np
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image

model = load('breastmodel3.joblib')


def show_predict_page():
    st.title("Now you can make your prediction.")
    img2 = Image.open('breast-cancer-symptoms2.png')
    st.image(img2)
    st.write("""The model predicts whether a tumor is benign or malignant, 
based on the attributes of the tumor shown below, which are on a scale of 1 to 10.""")
    st.write("Clump Thickness")
    st.write("Uniformity of Cell Size")
    st.write("Uniformity of Cell Shape")
    st.write("Marginal Adhesion")
    st.write("Single Epithelial Cell Size")
    st.write("Bare Nuclei")
    st.write("Bland Chromatin")
    st.write("Normal Nucleoli")
    st.write("Mitoses")
    st.subheader('The predictions made by the model are 97% accurate.')
    st.write(""" Use the slider to input the ratings of the tumor, then
    click to get what kind of tumor you're working with.""")
    st.write("Here are the performance metrics of the model:")
    st.write("Accuracy score: 0.9785714285714285")
    st.write("Precision score: 0.9797979797979798")
    st.write("Recall score: 0.9897959183673469")
    st.write("""Confusion matrix: [[97  1]
                                    [ 2 40]]""")
    st.write("F1 score: 0.9847715736040609")

    clmp_thk = st.slider("Rate the clump thickness.", 0, 10, 1)
    unicell = st.slider("Rate the uniformity of the cell size.", 0, 10, 1)
    unishape = st.slider("Rate the uniformity of the cell shape.", 0, 10, 1)
    margad = st.slider("Rate marginal adhesion of the tumor.", 0, 10, 1)
    singepi = st.slider("Rate the Single Epithelial cell size.", 0, 10, 1)
    barenuc = st.slider("Rate the Bare Nuclei.", 0, 10, 1)
    blachro = st.slider("Rate the Bland Chromatin.", 0, 10, 1)
    nornuc = st.slider("Rate the Normal Nucleoli.", 0, 10, 1)
    mitoses = st.slider("Rate the Mitoses.", 0, 10, 1)

    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.astype(float).values.tolist()
    df.drop(labels='id', axis=1, inplace=True)

    ko = st.button('So what kind of tumor is it?')
    if ko:
        x = np.array([clmp_thk, unicell, unishape, margad, singepi, barenuc, blachro, nornuc, mitoses])
        x = x.reshape(1, -1)
        res = model.predict(x)
        if res == [2]:
            st.write('This tumor is most likely a benign one.')
        else:
            st.write('This tumor is most likely a malignant one.')

