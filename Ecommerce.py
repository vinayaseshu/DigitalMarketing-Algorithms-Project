import streamlit as st
import pandas as pd
import plotly.express as px
from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session
import snowflake.connector as sf
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import plotly.express as px
import datetime

#Setting up connection with database
@st.cache_resource
def connect():
    
    CONNECTION_PARAMETERS = {
    "account": st.secrets['snowflake_acc'],
    "user": st.secrets['snowflake_user'],
    "password": st.secrets['snowflake_pass'],
        "database": st.secrets['snowflake_database'],
        "schema": st.secrets['snowflake_schema']
        
    }

    conn = sf.connect (
        user= st.secrets['snowflake_user'],
        password= st.secrets['snowflake_pass'],
        account= st.secrets['snowflake_acc'],
        warehouse= st.secrets['snowflake_warehouse'],
        database = st.secrets['snowflake_database'],
        schema=st.secrets['snowflake_schema']
    )

    session = Session.builder.configs(CONNECTION_PARAMETERS).create()

    engine = create_engine(URL(
        account = st.secrets['snowflake_acc'],
        user = st.secrets['snowflake_user'],
        password = st.secrets['snowflake_pass'],
        database = st.secrets['snowflake_database'],
        schema = st.secrets['snowflake_schema'],
        warehouse = st.secrets['snowflake_warehouse'],
        role='ACCOUNTADMIN',
    ))
    
    return conn, session, engine

#Caching data 
@st.cache_data
def get_data():
    hist_df = session.table('FORECASTING.DEMANDFORECASTING."Historical"').to_pandas()
    return hist_df
    
#Plotting data
def plot_hist(session, dist, prod, refresh=False):
    
    if refresh:
        with st.spinner('Training model with current data hold tight...'):
            query = "Call FORECASTXGBOOST_SPROC();"
            session.sql(f"{query}").collect()
        st.snow()
        st.success('Model trained successfully', icon="✅")
           
    hist_df = get_data()
    filtered_df = hist_df.loc[ (hist_df['product_distribution_center_id'] == dist) & (hist_df['product_category'] == prod)]
    filtered_df['date'] = pd.to_datetime(filtered_df['year'].astype(str) + '-' + filtered_df['month'].astype(str), format='%Y-%m')

    # Create a line graph using Plotly
    fig = px.line(filtered_df, x='date', y='count')
    fig.update_layout(width=800, height=400)
    # Set the title of the graph
    fig.update_layout(title=f'Trend of product category {prod} for distribution center {dist} over the years')
    fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales Quantity",
    font=dict(family="Arial", size=12, color="#7f7f7f"))


    # Display the graph on Streamlit
    st.plotly_chart(fig)
    
@st.cache_data
def create_futuredf(year, month):
    with st.spinner('Creating dataframe for making predictions...'):
        query = f"Call CREATE_FUTUREDFSF({year}, {month});"
        session.sql(f"{query}").collect()
    st.success('Predictions data set is ready call predict!', icon="✅")
    
    with st.spinner('Making predictions...'):
        df = pd.read_sql_query('Select *, FORECAST_XG(*) as Predictions from "FutureStock";', engine)
    st.success('Query Success', icon="✅")
    return df
    
    
    
def plot_pred(dist, df):
    
    hist_df = get_data()
    
    year = df['year'][0]
    month = df['month'][0]
    
    df_hist = hist_df[ (hist_df['product_distribution_center_id'] == dist) & (hist_df['year'] == year )  & (hist_df['month'] == month - 1)    ].copy()
    df_hist = df_hist[ ['year', 'month','product_category', 'count']].copy()
    df_hist['month'] = 'Previous Month'
    
    
    df_filtered = df[df['product_distribution_center_id'] == dist]
    df_filtered['product_category'] = df_filtered['product_category'].astype(str)
    df_filtered  = df_filtered.sort_values(by='predictions', ascending=False)
    df_filtered = df_filtered[ ['year', 'month','product_category', 'predictions']].copy()
    df_filtered.columns = ['year', 'month','product_category', 'count']
    df_filtered['month'] = 'Next Month'

    result = pd.concat([df_filtered, df_hist])
    
    result.month = result.month.astype(str)
    result.product_category = result.product_category.astype(str)


    # create the bar chart using Plotly Express
    fig = px.bar(result, x='product_category', y='count', color='month',barmode='group',
                title=f'Predictions for Center ID {dist} for the year {year} and month {month}')

    # display the chart
    st.plotly_chart(fig)
    
    
    
    
    
    
           

#Making connection
conn, session, engine = connect()


st.title("EcommAI: Demand Forecasting 🛍️🛒📈")
st.header('Historical trend of stock')
st.sidebar.success('Demand Forecasting')



refresh = st.button("refresh", key='refresh', help='Refresh will load the model with latest data from snowflake and train the model on the new data')
dist = int(st.selectbox('Select Distribution Center 🏭',[1,2,3,4,5,6,7,8,9,10],0 ))
prod = int(st.selectbox('Select Product category 👔👖👗', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],0))
   
if refresh:
    plot_hist(session=session, dist= dist, prod= prod, refresh=True)
        
plot = st.button('Plot', key='plot', help='This will plot the historical product stock sales')

if plot:
    plot_hist(session=session, dist= dist, prod= prod, refresh=False)

st.header('Forecasting for Future stocks 🔮')
st.markdown('Select next month as per current date for more accuracy. Accuracy decreases as you try to forecast more in future.')

user_year = int(st.number_input('Enter Year 📅', max_value=2024, step=1, min_value=2023, value=datetime.datetime.now().year))
user_month = int(st.number_input('Enter Month 🔢', max_value=12, step=1, min_value=1, value=datetime.datetime.now().month + 1))
dist_pred = int(st.selectbox('Select Distribution Center 🏭',[1,2,3,4,5,6,7,8,9,10],0, key='pred_select' ))

run = st.button('Run', key='run')

if run:
    dfpred = create_futuredf(year = user_year, month = user_month)
    st.write(dfpred)
    plot_pred(dist=dist_pred,df = dfpred)
            
