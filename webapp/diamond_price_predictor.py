import streamlit as st
import pickle
import numpy as np

def load_model(path_to_model):
    try:
        with open(path_to_model,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.sidebar.error(f'Model Loading error => {e}')


model_path = '../models/diamond_price_tree.pk'
model = load_model(model_path)

if model:
    st.sidebar.success("model loaded successfully")

st.title('Price Prediction for Diamonds')
st.subheader('by Zaid Kamil')

pressure = st.number_input('Enter pressure value')
palenoium = st.number_input('enter palenoium value')

btn =  st.button('Make price prediction')
if btn:
    data = np.array([[pressure,palenoium]])
    out = model.predict(data)
    st.success(f'Price {out[0]}')

# streamlit run diamond_price_predictor.py