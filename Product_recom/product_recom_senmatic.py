import pandas as pd
import numpy as np
import pinecone
import itertools
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st


def get_data():
    products = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/products.csv')
    return products

def get_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print(f"You are using {device}. This is much slower than using "
            "a CUDA-enabled GPU. If on Colab you can change this by "
            "clicking Runtime > Change runtime type > GPU.")
        
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    return model


def connect_pine(api,env):
    pinecone.init(api_key= api, environment= env)
    index_name = 'products-sim'
    index = pinecone.Index(index_name=index_name)
    return index



products = get_data()
# # Get unique values from a column in the DataFrame
options = products["name"].unique()


# Define Streamlit app
st.title("Associated Products")

# Display the selectbox widget
product = st.selectbox("Selected Product",options= options)

model = get_model()

api= '5d4c8961-5aaa-4aa5-a104-50561a86de58'
env='eu-west1-gcp'
index = connect_pine(api,env)

# create the query vector
vector = model.encode(product).tolist()
# now query
xc = index.query(vector, top_k=10, include_metadata=True)

df = pd.DataFrame({'Product': [result['id'] for result in xc['matches']],'score': [round(result['score'], 2) for result in xc['matches']]})
df = df[df['score'] != 1]

st.write(df)


