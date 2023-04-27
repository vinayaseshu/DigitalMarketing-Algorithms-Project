import pandas as pd
import numpy as np
import pinecone
import itertools
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st


# def get_data():
#     products = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/products.csv')
#     return products






data , products, city_item = get_data()
# # Get unique values from a column in the DataFrame
options = products["name"].unique()


# Define Streamlit app
st.title("Associated Products 🛒")

# Display the selectbox widget

product = st.selectbox("Selected Product",options= options)

model = get_asso_model()

# api= '5d4c8961-5aaa-4aa5-a104-50561a86de58'
# env='eu-west1-gcp'

api_asso = st.secrets['api_asso']
env_asso = st.secrets['env_asso']
index = connect_pine_asso(api_asso,env_asso)

# create the query vector
vector = model.encode(product).tolist()
# now query
xc = index.query(vector, top_k=10, include_metadata=True)

df = pd.DataFrame({'Product': [result['id'] for result in xc['matches']],'score': [round(result['score'], 2) for result in xc['matches']]})
df = df[df['score'] != 1]
st.subheader("Similar Products ✅")
st.write(df)


