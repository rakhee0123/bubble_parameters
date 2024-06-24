# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 23:10:20 2024

@author: Rakhee Prajapat
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# File paths
AR_model_path = 'C:/Users/91900/Documents/model_AR.sav'
DB_model_path = 'C:/Users/91900/Documents/model_DB.sav'
Deq_model_path = 'C:/Users/91900/Documents/model_Deq.sav'

# Load models
AR_model = pickle.load(open(AR_model_path, 'rb'))
DB_model = pickle.load(open(DB_model_path, 'rb'))
Deq_model = pickle.load(open(Deq_model_path, 'rb'))

# Normalization parameters (mean and std dev) - update these with your actual values
# Aspect Ratio model normalization parameters
AR_mean = np.array([95.317039, 27.402235, 41.387360, 0.505761])
AR_std = np.array([3.086010, 7.704188, 4.746471, 0.350716	])

# Base Diameter model normalization parameters
DB_mean = np.array([95.214511, 27.223975, 41.334543, 0.502760	])
DB_std = np.array([3.117997, 7.837557	, 4.876653, 0.355768	])

# Equivalent Diameter model normalization parameters
Deq_mean = np.array([95.269841	, 27.587302	, 41.312143, 0.498413])
Deq_std = np.array([3.220301	, 8.171387, 5.068016	, 0.352988])

# Function to normalize input data
def normalize(input_data, mean, std):
    return (input_data - mean) / std

# Function to predict the aspect ratio
def predict_aspect_ratio(input_data, model, mean, std):
    input_data = np.array(input_data).reshape(1, -1)
    normalized_data = normalize(input_data, mean, std)
    prediction = model.predict(normalized_data)
    return prediction[0, 0]

# Function to predict the base diameter
def predict_base_diameter(input_data, model, mean, std):
    input_data = np.array(input_data).reshape(1, -1)
    normalized_data = normalize(input_data, mean, std)
    prediction = model.predict(normalized_data)
    return prediction[0, 0]

# Function to predict the equivalent diameter
def predict_equivalent_diameter(input_data, model, mean, std):
    input_data = np.array(input_data).reshape(1, -1)
    normalized_data = normalize(input_data, mean, std)
    prediction = model.predict(normalized_data)
    return prediction[0, 0]

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Single vapor bubble parameters',
        ['Aspect ratio', 'Base diameter', 'Equivalent diameter'],
        icons=['hand-index', 'hand-index', 'hand-index'],
        default_index=0
    )

# Aspect ratio prediction page
if selected == 'Aspect ratio':
    st.title('Aspect ratio prediction using ML')
    col1, col2 = st.columns(2)
    
    with col1:
        subcooling = st.number_input('Enter the value of subcooling', value=0.0, format="%.2f")
        
    with col2:
        flow_rate = st.number_input('Enter the value of flow rate', value=0.0, format="%.2f")
    
    with col1:
        heat = st.number_input('Enter the value of heat', value=0.0, format="%.2f")
    
    with col2:
        normalized_time = st.number_input('Enter the value of normalized time', value=0.0, format="%.2f")
    
    user_input = [subcooling, flow_rate, heat, normalized_time]
    
    if st.button('Aspect ratio Result'):
        predicted_output = predict_aspect_ratio(user_input, AR_model, AR_mean, AR_std)
        st.success(f'Predicted Aspect Ratio: {predicted_output}')

# Base diameter prediction page
if selected == 'Base diameter':
    st.title('Base diameter prediction using ML')
    col1, col2 = st.columns(2)
    
    with col1:
        subcooling = st.number_input('Enter the value of subcooling', value=0.0, format="%.2f")
        
    with col2:
        flow_rate = st.number_input('Enter the value of flow rate', value=0.0, format="%.2f")
    
    with col1:
        heat = st.number_input('Enter the value of heat', value=0.0, format="%.2f")
    
    with col2:
        normalized_time = st.number_input('Enter the value of normalized time', value=0.0, format="%.2f")
    
    user_input = [subcooling, flow_rate, heat, normalized_time]
    
    if st.button('Base diameter Result'):
        predicted_output = predict_base_diameter(user_input, DB_model, DB_mean, DB_std)
        st.success(f'Predicted Base Diameter: {predicted_output}')

# Equivalent diameter prediction page
if selected == 'Equivalent diameter':
    st.title('Equivalent diameter prediction using ML')
    col1, col2 = st.columns(2)
    
    with col1:
        subcooling = st.number_input('Enter the value of subcooling', value=0.0, format="%.2f")
        
    with col2:
        flow_rate = st.number_input('Enter the value of flow rate', value=0.0, format="%.2f")
    
    with col1:
        heat = st.number_input('Enter the value of heat', value=0.0, format="%.2f")
    
    with col2:
        normalized_time = st.number_input('Enter the value of normalized time', value=0.0, format="%.2f")
    
    user_input = [subcooling, flow_rate, heat, normalized_time]
    
    if st.button('Equivalent diameter Result'):
        predicted_output = predict_equivalent_diameter(user_input, Deq_model, Deq_mean, Deq_std)
        st.success(f'Predicted Equivalent Diameter: {predicted_output}')
