import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.calculation.calc import Get_DataFrame_Info, Get_DataFrame_Value_info
from utils.calculation.chart import Get_histogram, Get_Density_Destributation, Get_Box_Plot
from utils.filter import Get_Numerical_Columns

from utils.core import Load_data
import os
from dotenv import load_dotenv
import json

load_dotenv(override=True)


def main_page():

    dataset = Load_data(os.getenv('DATASET_DIRR'))
    dataset.iloc[109] = dataset.iloc[109].fillna(0.9)
    st.title(os.getenv('TITLE_APP'))
    st.write("The whole cheatsheet:")

    st.write(dataset)

    st.write("Some statistics:")
    st.write(dataset.describe())

    st.divider()
    st.write("Some information:")
    st.table(Get_DataFrame_Info(dataset))

    st.divider()
    st.header("Column type:")
    typeOfFeatureDic = {x: eval(os.getenv(x))
                        for x in eval(os.getenv('COLUMN_TYPE'))}
    tabsOfFeatursTypes = zip(st.tabs(list(typeOfFeatureDic)), typeOfFeatureDic)
    for tab, typeofFeature in tabsOfFeatursTypes:
        with tab:
            tab.header(f'List of {typeofFeature} columns')
            st.table(Get_DataFrame_Value_info(
                dataset, typeOfFeatureDic[typeofFeature], typeofFeature))

    st.divider()
    st.header("Histogram:")
    listOfcolumnForHistogram = eval(os.getenv("HistogramPlot"))
    listoftab = zip(st.tabs(listOfcolumnForHistogram),
                    listOfcolumnForHistogram)
    for tab, featureName in listoftab:
        with tab:
            tab.header(f'{featureName}')
            fig, ax = Get_histogram(dataset, featureName)
            st.pyplot(fig)

    st.divider()
    st.header('Scatter Plot:')

    options = st.multiselect(
        'select two columns for makeing correlation plot',
        list(dataset.columns.values),
        max_selections=2
    )

    if len(options) == 2:
        st.scatter_chart(dataset, x=options[0], y=options[1])

    st.divider()
    st.header("Density plot:")
    optionDensity = st.selectbox(
        'What would you plot?',
        eval(os.getenv("DensityPlot")))
    if len(optionDensity):
        fig, ax = Get_Density_Destributation(dataset, optionDensity)
        st.pyplot(fig)

    numeric_data = Get_Numerical_Columns(dataset)
    st.divider()
    st.header("Box plot:")
    optionBoxPlot = st.selectbox(
        'What would you plot?',
        numeric_data)
    if len(optionDensity):
        # fig, ax = Get_Box_Plot(dataset, optionBoxPlot)
        fig = Get_Box_Plot(dataset, optionBoxPlot)
        st.pyplot(fig)
