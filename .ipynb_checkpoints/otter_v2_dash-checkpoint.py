# app.py

import streamlit as st
import pandas as pd
# import datetime

DATA_URL = "./otter_data.csv"

@st.cache
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

def calculate_metrics(filtered_data):
    # Compute the metrics for the filtered data here
    # For example:
    total_revenue = filtered_data['Revenue'].sum()
    average_daily_stickiness = (filtered_data['overall_daily_access'] / filtered_data['overall_daily_active']).mean()

    # Add more metrics as needed

    return {
        "Total Revenue": total_revenue,
        "Average Daily Stickiness": average_daily_stickiness
    }

df = load_data()

# Sidebar
st.sidebar.header('Filters')

# Region Filter
regions = df['region'].unique().tolist()
selected_region = st.sidebar.multiselect('Select Region', regions, regions)

# Country Filter
countries = df[df['region'].isin(selected_region)]['Country'].unique().tolist()
selected_country = st.sidebar.multiselect('Select Country', countries, countries)

# Time Range Filter
df['day_partition'] = pd.to_datetime(df['day_partition'])
min_date = df['day_partition'].min()
max_date = df['day_partition'].max()
date_range = st.sidebar.slider('Select Date Range', min_date, max_date, (min_date, max_date))

# Filter data based on user input
filtered_data = df[(df['region'].isin(selected_region)) &
                   (df['Country'].isin(selected_country)) &
                   (df['day_partition'].between(date_range[0], date_range[1]))]

# Calculate Metrics
metrics = calculate_metrics(filtered_data)

# Display Metrics
st.header("Metrics")
for metric, value in metrics.items():
    st.subheader(f"{metric}: {value}")

# Optionally Display Filtered Data
if st.checkbox('Show Filtered Data'):
    st.write(filtered_data)

