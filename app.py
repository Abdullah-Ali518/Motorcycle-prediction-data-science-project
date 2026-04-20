import streamlit as st
import pandas as pd
import pickle
import numpy as np
try:
    model=pickle.load(open('bike_model (2).pkl','rb'))
    scaler=pickle.load(open('scaler (2).pkl ','rb'))
except Exception as e:
    st.error(f"Error loading files:{e}")
st.set_page_config(page_title="Bike Price Predictor",layout="centered")
st.title("Motorcycle Price prediction")
year=st.number_input("Year of purchase",1990,2026,2020)
km_driven=st.number_input("kilometers Driven",0,500000,10000)
ex_showroom=st.number_input("Ex-showroom Price (Rs)",0,2000000,80000)
brand=st.number_input("Brand Code (e.g., Honda=2,RE=11)",0,50,0)
seller_type=st.selectbox("Seller Type",options=[0,1],format_func=lambda x: "Individual" if x==1 else "Dealer")
owner=st.selectbox("Owner Type",options=[0,1,2,3],format_func=lambda x: f"{x+1}st Owner") 
if st.button("Predict Price"):
    input_data=np.array([[year,seller_type,owner,km_driven,ex_showroom,brand]])
    input_scaled=scaler.transform(input_data)
    prediction=model.predict(input_scaled) 
    st.success(f"### Estimated Selling Price Rs. {int(prediction[0]):}")        