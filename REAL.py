
import streamlit as st
import pandas as pd
import joblib
import model
from PIL import Image
#app title
st.header("REAL: The Real Estate Prediction Tool")
st.markdown("Welcome to REAL, The Real Estate Prediction Tool. To receive a price prediction for a property, please enter the information requested below:")
#inputs
bed= st.number_input("Enter the number of bedrooms, between 1 and 15:", min_value=1, max_value=15)
bath = st.number_input("Enter the number of bathrooms, between 1 and 20:", min_value = 1, max_value=20)
acre_lot= st.number_input("Enter the number of acres, between 0 and 100:", min_value= 0.0, max_value = 100.00)
house_size= st.number_input("Enter the number of square feet of the property (without commas):", min_value =100, max_value =150000)
zip_code = st.selectbox("Select the property zip code from the list of available zip codes:",  model.zip_codes)

#If button is pressed, use input to give prediction.
if st.button("Submit"):
#Unpickle the classifier
    model.model = joblib.load("model.pkl")
    #Store user input into df
    new_input_df = pd.DataFrame([[bed,bath,acre_lot,zip_code,house_size]], columns=["bed", "bath", "acre_lot", "zip_code", "house_size"])
    new_prediction_df= model.model.predict(new_input_df)
    new_prediction_int = int(new_prediction_df.round())
    new_prediction_int= "{:,}".format(new_prediction_int)


    #Give price prediction
    st.markdown(f'A house in the zip code {zip_code} with {bed} bedroom(s), {bath} bathroom(s),  {acre_lot} acre(s), and {house_size} square feet would be approximately ${new_prediction_int}.')
    #Give r^2
#     r2 = model.model.score(model.X_test, model.y_test)
#     st.markdown("Prediction r^2 is r2")


#posting the graphs

price_graph = Image.open('price-graph.png')
st.image(price_graph)

size_graph= Image.open('size-graph.png')
st.image(size_graph)

pie= Image.open('pie_chart.png')
st.image(pie)
