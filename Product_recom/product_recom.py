import pandas as pd
import numpy as np
import requests
import io
import pinecone
import streamlit as st
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def get_data():
    #Master Data
    data = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/Data_Prep/master_data.csv')

    #Products data
    products = pd.read_csv("https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Data/products.csv")
    products = pd.read_csv("https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Data/products.csv")
    products = products.dropna(subset=['name'])
    products = products.rename(columns={'id': 'product_id', 'name': 'product_name'})
    products = products[['product_id', 'category','product_name', 'brand', 'department']]

    #city_item Data
    city_item = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/Data_Prep/master_data.csv')
    city_item = city_item[['city','product_id','Quantity']]

    return data, products, city_item

@st.cache_data
def get_factors():
    #Model 1factors
    url = "https://github.com/ashwinkadam/DigitalMarketing-Algorithms-Project/blob/main/Product_recom/Models/Model_1/item_factors.npy?raw=true"
    r = requests.get(url).content
    file = io.BytesIO(r)
    item_factors_m1 = np.load(file)

    #Model 2 factors
    url = "https://github.com/ashwinkadam/DigitalMarketing-Algorithms-Project/blob/main/Product_recom/Models/Model_2/item_factors.npy?raw=true"
    r = requests.get(url).content
    file = io.BytesIO(r)
    item_factors_m2 = np.load(file)

    #Model 1factors
    url = "https://github.com/ashwinkadam/DigitalMarketing-Algorithms-Project/blob/main/Product_recom/Models/Model_3/age_factors.npy?raw=true"
    r = requests.get(url).content
    file = io.BytesIO(r)
    item_factors_m3 = np.load(file)

    return item_factors_m1, item_factors_m2, item_factors_m3


def get_index():
    data , products, city_item = get_data()

    #Model 1 Index creation
    users = list(np.sort(data.user_id.unique()))
    index_to_user = pd.Series(users)
    user_to_index = pd.Series(data=index_to_user.index , index=index_to_user.values)

    #Model 2 Index creation
    city = list(np.sort(data.city.unique()))
    index_to_city = pd.Series(city)
    city_to_index = pd.Series(data=index_to_city.index , index=index_to_city.values)

    #Indexing 
    age = list(np.sort(data.age.unique()))
    index_to_age= pd.Series(age)
    age_to_index = pd.Series(data=index_to_age.index , index=index_to_age.values)

    return user_to_index,city_to_index,age_to_index

@st.cache_resource
def pinecone_1(api,env):
    # Initialize Pinecone and load data
    pinecone.init(api_key=api, environment=env)
    # pinecone.init(api_key='863bc7fa-db48-4cc1-9fbe-050e92ee681e', environment='us-west4-gcp')
    index_name = 'user-products'
    index_1 = pinecone.Index(index_name=index_name)
    return index_1


def pinecone_2(api,env):
    # Initialize Pinecone and load data
    pinecone.init(api_key=api, environment=env)
    # pinecone.init(api_key= '32ae2c93-dc47-4452-b6d1-51329bb62822', environment='northamerica-northeast1-gcp')
    index_name = 'city-products'
    index_2 = pinecone.Index(index_name=index_name)
    return index_2


def pinecone_3(api,env):
    # Initialize Pinecone and load data
    pinecone.init(api_key=api, environment=env)
    # pinecone.init(api_key= '2e8bb60b-fe4e-4b5a-b4ff-c4dc103edd39', environment='asia-northeast1-gcp')
    index_name = 'age-products'
    index_3 = pinecone.Index(index_name=index_name)
    return index_3

##################################### HELPER FUNCTION ###################################################

# Function to get products bought by user in the past
def products_bought_by_user_in_the_past(user_id, top):
    data , products, city_item = get_data()
    selected = data[data.user_id == user_id].sort_values(by=['Quantity'], ascending=False)
    selected['product_name'] = selected['product_id'].map(products.set_index('product_id')['product_name'])
    selected = selected[['product_id', 'product_name', 'Quantity']].reset_index(drop=True)
    selected['category'] = selected['product_id'].map(products.set_index('product_id')['category'])
    selected['department'] = selected['product_id'].map(products.set_index('product_id')['department'])
    selected = selected[['product_name', 'category','department']]

    if selected.shape[0] < top:
        return selected
    return selected[:top]


def create_lookup_city_df():
    data , products, city_item = get_data()
    # Convert categorical columns to numerical codes
    le = LabelEncoder()
    city_item_conv = city_item.copy()
    for column in city_item.select_dtypes(include=['object']):
        city_item_conv[column] = le.fit_transform(city_item_conv[column])
    
    # Create lookup table mapping city names to numerical codes
    lookup_city = pd.merge(city_item, city_item_conv, left_index=True, right_index=True)
    lookup_city = lookup_city[['city_x', 'city_y']].copy()
    return lookup_city

def check_user(user_id):
    data = pd.read_csv('https://raw.githubusercontent.com/ashwinkadam/DigitalMarketing-Algorithms-Project/main/Product_recom/Data_Prep/master_data.csv')
    if user_id in [data[data['user_id'] == 5]['user_id'].unique()[0]]:
        return True
    else:
        return False
     
##################################### MODEL 1 : USER_ITEM ###################################################


def get_recommendations_Model_1(user_id, item_factors, user_to_index,index):

    data , products, city_item = get_data()

    # Generate recommendations
    user_factors = item_factors[user_to_index[user_id]]
    query_results = index.query(queries=[user_factors.tolist()], top_k=5)


    # Display recommendations and top buys from the past
    if query_results.results:
        for _id, res in zip([user_id], query_results.results):
            p = [match.id for match in res.matches]
            s = [match.score for match in res.matches]
            df = pd.DataFrame(
                {
                    'product_name': p,
                    'scores': s
                }
            )
            df = df[df['scores'] > 0.5]
            df = df.merge(products, on='product_name', how='inner')[['product_name', 'category', 'department', 'scores']]
            dep = list(products_bought_by_user_in_the_past(user_id, top=10)['department'])[0]
            df = df[df['department'] == dep]
        top_buys_df = products_bought_by_user_in_the_past(user_id, top=10)
        return df, top_buys_df
    else:
        return None, None
    

###################################### MODEL 2 : CITY_ITEM ###################################################

lookup_city = create_lookup_city_df()

def get_recommendations_Model_2(user_id, item_factors, city_to_index,index):

    data , products, city_item = get_data()

    # Generate recommendations
    city_id = lookup_city[lookup_city['city_x'] == data[data['user_id'] == 5]['city'].unique()[0]]["city_y"].unique()[0]

    user_factors = item_factors[city_to_index[city_id]]
    query_results = index.query(queries=[user_factors.tolist()], top_k=5)


    # Display recommendations and top buys from the past
    if query_results.results:
        for _id, res in zip([city_id ], query_results.results):
            p = [match.id for match in res.matches]
            s = [match.score for match in res.matches]
            df = pd.DataFrame(
                {
                    'product_name': p,
                    'scores': s
                }
            )
            df = df[df['scores'] > 0.5]
            df_city = df.merge(products, on='product_name', how='inner')[['product_name', 'category', 'department', 'scores']]

        return df_city
    else:
        return None, None


##################################### MODEL 3 : AGE_ITEM ###################################################

def get_recommendations_Model_3(user_id, item_factors, city_to_index,index):

    data , products, city_item = get_data()

    # Generate recommendations
    age_id = data[data['user_id'] == 5]['age'].unique()[0]

    user_factors = item_factors[city_to_index[age_id]]
    query_results = index.query(queries=[user_factors.tolist()], top_k=5)


    # Display recommendations and top buys from the past
    if query_results.results:
        for _id, res in zip([age_id], query_results.results):
            p = [match.id for match in res.matches]
            s = [match.score for match in res.matches]
            df = pd.DataFrame(
                {
                    'product_name': p,
                    'scores': s
                }
            )
            df = df[df['scores'] > 0.5]
            df_age = df.merge(products, on='product_name', how='inner')[['product_name', 'category', 'department', 'scores']]
        return df_age
    else:
        return None, None
    

##################################### MAIN #################################################################

#api_1 = '863bc7fa-db48-4cc1-9fbe-050e92ee681e'
#env_1 = 'us-west4-gcp'
api_1 = st.secrets['api_1']
env_1 = st.secrets['env_1']
index_1 = pinecone_1(api_1,env_1)

# api_2 = '32ae2c93-dc47-4452-b6d1-51329bb62822'
# env_2 = 'northamerica-northeast1-gcp'
api_2 = st.secrets['api_2']
env_2 = st.secrets['env_2']
index_2 = pinecone_2(api_2,env_2)

# api_3 = '2e8bb60b-fe4e-4b5a-b4ff-c4dc103edd39'
# env_3 = 'asia-northeast1-gcp'
api_3 = st.secrets['api_3']
env_3 = st.secrets['env_3']
index_3 = pinecone_3(api_3,env_3)

item_factors_m1, item_factors_m2, item_factors_m3 = get_factors()

user_to_index , city_to_index , age_to_index = get_index()


# Streamlit app code
st.title("Product Recommendation")
# Get user input
user_id = int(st.text_input("Enter user ID:", value=5))

if check_user(user_id):
    #Model 1
    df, top_buys_df = get_recommendations_Model_1(user_id, item_factors_m1, user_to_index,index_1)
    is_empty = df.empty
    if is_empty:
         st.subheader('User has less Interaction - No Product Recommendation for now based on User Interaction')
    else:
        st.subheader("Recommendations based on User Interaction:")
        st.write(df)

    #Model 2
    df_city = get_recommendations_Model_2(user_id,item_factors_m2,city_to_index,index_2)
    st.subheader("Recommendations based on Region:")
    st.write(df_city)

    #Model 3
    df_age = get_recommendations_Model_3(user_id,item_factors_m3, age_to_index,index_3)
    st.subheader("Recommendations based Age:")
    st.write(df_age)

    #Past buy's
    st.subheader("Top buys from the past:")
    st.write(top_buys_df)
else:
    print("Invalid User")

