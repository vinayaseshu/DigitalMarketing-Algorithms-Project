# EcommAI: Empowering Online Shopping with Intelligent Tools

# Problem Statement and Solution

Company ABC is struggling to manage their inventory stocks, predict future demand, and provide an enhanced customer journey on their website. With a small customer base, they need a solution that can provide insights to drive business decisions and improve customer satisfaction.

We, NEU-Analytics, propose the following solutions to address the challenges faced by Company ABC:

1. Build a demand forecasting model to predict future demand
2. Implement semantic search for products to ease customer journey from landing to checkout
3. Build a knowledge base based on YouTube review videos to perform generative QA
4. Build a product recommendation model to implement hybrid recommendations

# Architecture: 

## 1. Demand Forecasting 
![MicrosoftTeams-image (13)](https://user-images.githubusercontent.com/81140802/235243695-1dde04cf-b29f-4fda-bb7b-3016cc76cd89.png)
## 2. Recommendation System
![MicrosoftTeams-image (11)](https://user-images.githubusercontent.com/81140802/235243753-766a1bb6-581e-4d2b-ac0b-146fa457d033.png)
## 3. QA System
![QA system drawio](https://user-images.githubusercontent.com/81140802/235237065-2273bbe5-c0d8-4f4e-9149-17ccdee88152.png)

# Installation
To install the dependencies for this project, please run the following command:
```
pip install -r requirements.txt
```
# Usage
To run the project, please follow these steps:

1. Clone the project repository
2. Run the Demand Forecasting notebook to (Setup snowflake environment, Transform and load data tables onto snowflake, Deploy Model, UDFs & Stored Proc)
3. Run Product_Associate notebook to (Setup pinecone environment, Transform data and train model to upload vector embeddings onto pinecone index)
4. Launch Ecommerce.py file using 
```
Streamlit run Ecommerce.py
```


# Project Report
[EcommAI - Project Report](https://codelabs-preview.appspot.com/?file_id=1sgUkcEJZG9F1Q--qwzl0l6zqqxJByvsBNGrl1tGYbUg#0)

# Streamlit App
[EcommAI - Streamlit App](https://ecommai.streamlit.app/)

# Video Demonstration
[EcommAI - Video](https://youtu.be/SUTK-MF-YaE)
