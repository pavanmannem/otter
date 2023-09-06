# app.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import warnings 

warnings.filterwarnings('ignore')

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
# df['day_partition'] = pd.to_datetime(df['day_partition'])
# print("------------------------------------------")
# print(df.dtypes)

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
min_date = df['day_partition'].min().to_pydatetime()
max_date = df['day_partition'].max().to_pydatetime()

print(min_date, type(min_date), max_date, type(max_date))

selected_date = st.slider(
    "Select a date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    step=timedelta(days=1),
)
# date_range = st.sidebar.slider('Select Date Range', min_date, max_date, (min_date, max_date))

# Filter data based on user input
filtered_data = df[(df['region'].isin(selected_region)) &
                   (df['Country'].isin(selected_country)) &
                   (df['day_partition'].between(selected_date[0], selected_date[1]))]

# Calculate Metrics
metrics = calculate_metrics(filtered_data)

# Display Metrics
st.header("Metrics")
for metric, value in metrics.items():
    st.subheader(f"{metric}: {value}")

# Optionally Display Filtered Data
if st.checkbox('Show Filtered Data'):
    st.write(filtered_data)

if __name__ == "__main__":
    st.run()
