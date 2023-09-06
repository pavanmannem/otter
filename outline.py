import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

color_sequence = px.colors.qualitative.Plotly
st.set_page_config(layout="wide") 


## Helper Functions
def get_dau_mau(daily, monthly):
    if daily == 0:
        return 0
    else:
        return daily / monthly * 100


def get_dau_wau(daily, weekly):
    if daily == 0:
        return 0
    else:
        return daily / weekly * 100
    

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://theme.zdassets.com/theme_assets/9135151/75332ad7b3dc3cf77b34552a452fdaa9b2d2cf40.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Otter";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()
# Sample data load (you'd replace this with your own data loading mechanism)
df = pd.read_csv('otter_data.csv')

# Assuming df is loaded
st.title("Product Analytics Dashboard ")
# st.subheader("")


# logo_url = "logo.png"


# Sidebar granularity selection
granularity = st.sidebar.radio("Choose Granularity", ["Overall", "Regional", "Country"], )


# Convert 'day_partition' from string to datetime
df['day_partition'] = pd.to_datetime(df['day_partition'])
min_date = df['day_partition'].min().to_pydatetime()
max_date = df['day_partition'].max().to_pydatetime()
selected_date = st.slider(
    "Select a date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    step=timedelta(days=1),
)

# Filter data by date range
filtered_df = df[df['day_partition'].between(selected_date[0], selected_date[1])]

if granularity == "Regional":
    regions = ['ALL'] + list(filtered_df["region"].unique())
    selected_regions = st.multiselect("Select Regions", regions, default="ALL")

    if "ALL" not in selected_regions:
        filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
        
elif granularity == "Country":
    countries = ['ALL'] + list(filtered_df["country"].unique())
    selected_countries = st.multiselect("Select Countries", countries, default="ALL")

    if "ALL" not in selected_countries:
        filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]


products = ['overall','premium' ,'core','promos','custom_websites','basic_insights','adv_insights','super_insights']

for product in products:
    print(product, "started")
    filtered_df[product + '_dau_mau'] = filtered_df.apply(lambda x: get_dau_mau(x[product + "_daily_active"], x[product + "_monthly_active"]), axis = 1)
    filtered_df[product + '_dau_wau'] = filtered_df.apply(lambda x: get_dau_wau(x[product + "_daily_active"], x[product + "_weekly_active"]), axis = 1)

# Subsection: Product Adoption
st.subheader("Product Adoption")
col1, col2 = st.columns(2)
with col1:
    st.write("Product Adoption Plot 1")
with col2:
    st.write("Product Adoption Plot 2")

st.markdown('#')

# Subsection: Stickiness
st.subheader("Stickiness")

fig = px.line(filtered_df, 
            x="day_partition", 
            y=[col for col in filtered_df.columns if "_dau_mau" in col], 
            title='Product Stickiness', 
            color_discrete_sequence=color_sequence, 
            height=600,
            width=1500)

fig.update_xaxes(title="Date")
fig.update_yaxes(title="DAU / MAU (%)")
st.plotly_chart(fig)

# col1, col2 = st.columns(2)
# with col1:
#     st.write("Stickiness Plot 1")
# with col2:
#     st.write("Stickiness Plot 2")

# col3, col4 = st.columns(2)
# with col3:
#     st.write("Stickiness Plot 3")
# with col4:
#     st.write("Stickiness Plot 4")

st.markdown('#')

# Subsection: Revenue
st.subheader("Revenue")
col1, col2 = st.columns(2)
with col1:
    st.write("Revenue Plot 1")
with col2:
    st.write("Revenue Plot 2")

col3, col4 = st.columns(2)
with col3:
    st.write("Revenue Plot 3")
with col4:
    st.write("Revenue Plot 4")





st.markdown('#')


st.subheader("Raw Filtered Data")
# Display the data table (optional)
st.write(filtered_df)