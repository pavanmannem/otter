import pandas as pd 
import datetime 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams

# rcParams['font.family'] = 'Public Sans'

df = pd.read_csv('./otter_data.csv')


# Assuming you've loaded your dataframe as 'df' and your revenue table as 'rev_table'
# Also, you'll need to extract and calculate fees from the rev_table. This example assumes constant values.

def calculate_revenue(row):
    # Extracting user counts
    premium_users = row['premium_monthly_active']
    core_users = row['core_monthly_active']
    promos_users = row['promos_monthly_active']
    websites_users = row['custom_websites_monthly_active']
    basic_insights_users = row['basic_insights_monthly_active']
    adv_insights_users = row['adv_insights_monthly_active']
    super_insights_users = row['super_insights_monthly_active']

    # Replace with actual values from your revenue table
    premium_fee = 20  # monthly
    premium_activation = 100
    core_fee = 15  # monthly
    promos_fee = 20  # monthly
    websites_fee = 5  # monthly
    basic_insights_fee = 5  # monthly
    adv_insights_fee = 10 / 12  # yearly fee divided by 12 for monthly value
    super_insights_fee = 20  # once off, assume charged once for this example

    # Calculating revenue
    premium_revenue = premium_users * (premium_fee + premium_activation)
    core_revenue = core_users * core_fee
    promos_revenue = promos_users * promos_fee
    websites_revenue = websites_users * websites_fee
    basic_insights_revenue = basic_insights_users * basic_insights_fee
    adv_insights_revenue = adv_insights_users * adv_insights_fee
    super_insights_revenue = super_insights_users * super_insights_fee

    return pd.Series([premium_revenue, core_revenue, promos_revenue, websites_revenue, basic_insights_revenue, adv_insights_revenue, super_insights_revenue])

# Apply revenue calculation
df[['premium_revenue', 'core_revenue', 'promos_revenue', 'websites_revenue', 'basic_insights_revenue', 'adv_insights_revenue', 'super_insights_revenue']] = df.apply(calculate_revenue, axis=1)

# Convert day_partition to datetime and resample by quarter
df['day_partition'] = pd.to_datetime(df['day_partition'])
quarterly_revenues = df.resample('Q', on='day_partition').sum()[['premium_revenue', 'core_revenue', 'promos_revenue', 'websites_revenue', 'basic_insights_revenue', 'adv_insights_revenue', 'super_insights_revenue']]

# Streamlit option for monthly or quarterly view
view_option = st.selectbox("Choose Time View:", ['Monthly', 'Quarterly'])

# Resample data based on user choice
if view_option == 'Monthly':
    resampled_revenues = df.resample('M', on='day_partition').sum()[['premium_revenue', 'core_revenue', 'promos_revenue', 'websites_revenue', 'basic_insights_revenue', 'adv_insights_revenue', 'super_insights_revenue']]
    ticks = resampled_revenues.index.month
else:
    resampled_revenues = df.resample('Q', on='day_partition').sum()[['premium_revenue', 'core_revenue', 'promos_revenue', 'websites_revenue', 'basic_insights_revenue', 'adv_insights_revenue', 'super_insights_revenue']]
    ticks = resampled_revenues.index

# Streamlit code to display the line chart using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
resampled_revenues.plot(ax=ax)

# Add vertical lines
for date in resampled_revenues.index:
    ax.axvline(x=date, linestyle='--', color='grey', alpha=0.5)

# Formatting and displaying in Streamlit
ax.set_title(f'{view_option} Revenue Growth')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue')
ax.set_xticks(ticks)
st.pyplot(fig)