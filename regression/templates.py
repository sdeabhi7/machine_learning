import os 
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def save_model(obj,path_to_save):
    with open(path_to_save, 'wb') as f:
        pickle.dump(obj, f)
    return True

def load_model(path_to_model):
    with open(path_to_model,'rb') as f:
        return pickle.load(f)