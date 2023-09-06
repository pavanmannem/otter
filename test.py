import streamlit as st
import pandas as pd
import datetime

# Sample data
data = pd.read_csv('otter_data.csv')

data['day_partition'] = pd.to_datetime(data['day_partition']).dt.date

# Sidebar for Filters
st.sidebar.header('Filters')

# Time Filter
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 1, 1))
if start_date > end_date:
    st.sidebar.error('Error: End Date must fall after Start Date.')

# Geographical Filters
regions = data['region'].unique().tolist()
selected_region = st.sidebar.selectbox("Select Region", regions)
countries = data[data['region'] == selected_region]['Country'].unique().tolist()
selected_country = st.sidebar.multiselect("Select Countries", countries)

# Product Filter
# products = ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5', 'Product 6', 'Product 7']
# selected_product = st.sidebar.multiselect("Select Products", products)

# Filtering the Data
filtered_data = data[
    (data['day_partition'] >= start_date) & 
    (data['day_partition'] <= end_date) & 
    (data['region'] == selected_region) & 
    (data['Country'].isin(selected_country)) 
]

# Visualizing the Filtered Data
# This is just a placeholder. You would typically have charts/graphs here.
st.write(filtered_data)

# Reset Filters (Clears the cache and reloads the page)
if st.sidebar.button("Clear Filters"):
    st.experimental_rerun()