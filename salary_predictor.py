import streamlit as st
import os 
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
from regression.templates import load_model
from sklearn.preprocessing import PolynomialFeatures
# command to run, on cmd
# streamlit run salary_predictor.py

st.title('Postion wise Salary prediction')

job_titles =['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager',
 'Country Manager', 'Region Manager', 'Partner', 'Senior Partner','C-level', 'CEO']

job_title = st.selectbox('Please select a job title', job_titles)
btn = st.button('Show predicted salary')

if job_title and btn:
    model = load_model('models/position_salary.pk')
    st.write(model)
    pf = model.get('polynomial')
    reg = model.get('reg')
    pos =job_titles.index(job_title) + 1
    x = np.array([[pos]])
    px = pf.transform(x)
    result = reg.predict(px)
    st.text(f'your salary for {job_title}')
    st.success(round(result[0]))