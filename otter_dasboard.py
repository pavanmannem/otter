import streamlit as st
import pandas as pd
import calendar
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide") 

# Helper Functions ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pricing_structure = {
    'premium': {
        'Region 1': (20, 100),
        'Region 2': (30, 150),
        'Region 3': (80, 200)
    },
    'core': {
        'Region 1': (15, 0),
        'Region 2': (30, 0),
        'Region 3': (60, 0)
    },
    'promos': {
        'Region 1': (20, 0),
        'Region 2': (15, 0),
        'Region 3': (80, 0)
    },
    'custom_websites': {
        'Region 1': (5, 0),
        'Region 2': (5, 0),
        'Region 3': (20, 0)
    },
    'basic_insights': {
        'Region 1': (5, 0),
        'Region 2': (5, 0),
        'Region 3': (10, 0)
    },
    'adv_insights': {
        'Region 1': (10/12, 0),  
        'Region 2': (10/12, 0),
        'Region 3': (60/12, 0)
    },
    'super_insights': {
        'Region 1': (0, 20),
        'Region 2': (0, 25),
        'Region 3': (0, 30)
    }
}

 # Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('latest_otter_data.csv')
    df.columns = map(str.lower, df.columns)
    df.columns = map(str.strip, df.columns)

    df['day_partition'] = pd.to_datetime(df['day_partition'])
    df['month_year_period'] = df['day_partition'].dt.to_period('M')
    return df   


# Function to format revenue
def format_revenue(revenue):
    return f"{revenue/1e6:.1f}M"

def get_revenueue_table(df):
    revenue_df = df.copy()
    revenue_df = revenue_df[['day_partition','region','country'] + [col for col in revenue_df.columns if 'monthly_access' in col]]

    revenue_df['month'] = revenue_df['day_partition'].dt.month
    revenue_df['year'] = revenue_df['day_partition'].dt.year

    revenues = []
    for product in pricing_structure:
        for region, (monthly_fee, activation_fee) in pricing_structure[product].items():
            for country in revenue_df['country'].unique():
                subset_data = revenue_df[(revenue_df['region'] == region) & (revenue_df['country'] == country)]
                
                max_access_df = subset_data.groupby(['year', 'month']).agg(max_access=pd.NamedAgg(column=f'{product}_monthly_access', aggfunc='max')).reset_index()
                previous_max_access_df = max_access_df.copy()
                previous_max_access_df['month'] = previous_max_access_df['month'] - 1
                previous_max_access_df.columns = ['year', 'month', 'previous_max_access']
                
                monthly_data = pd.merge(max_access_df, previous_max_access_df, on=['year', 'month'], how='left').fillna(0)

                monthly_data['new_users'] = monthly_data['max_access'] - monthly_data['previous_max_access']
                monthly_data['new_users'] = monthly_data['new_users'].clip(lower=0)  # No negative new users

                if product == "super_insights":
                    monthly_data['revenue'] = monthly_data['new_users'] * activation_fee
                elif product == "adv_insights":
                    monthly_data['revenue'] = monthly_data['max_access'] * monthly_fee
                else:
                    monthly_data['revenue'] = monthly_data['max_access'] * monthly_fee + monthly_data['new_users'] * activation_fee

                for index, row in monthly_data.iterrows():
                    revenues.append({
                        'region': region,
                        'country': country,
                        'product': product,
                        'year': row['year'],
                        'month': row['month'],
                        'revenue': row['revenue']
                    })

    final_revenue_df = pd.DataFrame(revenues)



    # Convert year and month columns to a single column with format 'YYYY Mon'
    final_revenue_df['year'] = final_revenue_df['year'].astype(int)  # Ensure year is int
    final_revenue_df['month'] = final_revenue_df['month'].astype(int)  # Ensure month is int
    final_revenue_df['year_month'] = final_revenue_df['year'].astype(str) + ' ' + final_revenue_df['month'].apply(lambda x: calendar.month_abbr[x])

    # Drop the separate 'year' and 'month' columns
    final_revenue_df.drop(columns=['year', 'month'], inplace=True)
    return final_revenue_df

def plot_total_revenue(final_revenue_df):

    pivot_df = final_revenue_df.pivot_table(index=['year_month'], columns='region', values='revenue', aggfunc='sum').fillna(0).reset_index()

    color_map = {
        'Region 1': '#1F7368',
        'Region 2': '#B185B4',
        'Region 3': '#A6C1D8'
    }

    fig = go.Figure()

    for region in pivot_df.columns[1:]:
        fig.add_trace(go.Bar(
            x=pivot_df['year_month'],
            y=pivot_df[region],
            name=region,
            marker_color=color_map[region]
        ))

 
    fig.update_layout(
        barmode='stack',
        xaxis_title='Date',
        yaxis_title='Revenue $USD',
        xaxis={'categoryorder': 'array', 'categoryarray': final_revenue_df['year_month'].drop_duplicates().tolist()},
        width=1500,
        height=800,
        legend=dict(
            x=0.89,
            y=0.9
        ),
        legend_traceorder='normal',
            font=dict(
        color="#242526"  
    )
    )



    for i, date in enumerate(pivot_df['year_month']):
        total_revenue_for_month = sum([pivot_df[region].iloc[i] for region in pivot_df.columns[1:]])
        running_total = 0  # to track the stacked bar height for placing annotations
        for region in pivot_df.columns[1:]:
            segment_revenue = pivot_df[region].iloc[i]
            percentage = segment_revenue/total_revenue_for_month * 100
            fig.add_annotation(
                x=date,
                y=running_total + segment_revenue/2,
                text=f"<b>{format_revenue(segment_revenue)}<b><br><b>({percentage:.0f}%)</b>",
                showarrow=False
            )
            running_total += segment_revenue

    # Show the figure
    st.plotly_chart(fig)

def plot_product_revenue(final_revenue_df):

    final_revenue_df['year_month'] = pd.to_datetime(final_revenue_df['year_month'])
    pivot_df = final_revenue_df.pivot_table(index='year_month', columns='product', values='revenue', aggfunc='sum').fillna(0).reset_index()
    pivot_df = pivot_df.sort_values(by='year_month')

    color_map = {
        'premium': '#1F7368',
        'core': '#B185B4',
        'promos': '#A6C1D8',
        'custom_websites': '#FF7F0E',
        'basic_insights': '#9467BD',
        'adv_insights': '#2CA02C',
        'super_insights': '#17BECF'
    }

    fig = go.Figure()

    # 1. Calculate the total revenue for each year_month
    pivot_df['total_revenue'] = pivot_df.iloc[:, 1:].sum(axis=1)


    # Loop for each product
    for product in pivot_df.columns[1:-2]:  # Exclude 'year_month', 'total_revenue', and 'cumulative'
        product_revenue = pivot_df[product]
        percentage = (product_revenue / pivot_df['total_revenue']) * 100
        
        # Generate text labels
        text_labels = [f"<b>${val/1e6:.1f}M<br>({pct:.0f}%)<b>" for val, pct in zip(product_revenue, percentage)]

        
        # 4. Add traces with labels and determine positions
        fig.add_trace(go.Bar(
            x=pivot_df['year_month'],
            y=product_revenue,
            name=product,
            marker_color=color_map.get(product, 'grey'),
            text=text_labels,
            textposition='inside',  # Place text inside the bars
            insidetextanchor='middle'
        ))

    fig.update_layout(
        barmode='stack',  
        xaxis_title='Date',
        yaxis_title='Revenue $USD',
        width=1500,
        height=800,
        xaxis={
            'type': 'category',
            'tickmode': 'array',
            'tickvals': pivot_df['year_month'],
            'ticktext': pivot_df['year_month'].dt.strftime('%Y %b')
        },        
        legend=dict(
            x=0.89,
            y=0.9
        )
    )

    # Show the chart

    st.plotly_chart(fig)


def determine_color(value):
    return "green" if value > 0 else "red"


def plot_revenue_kpis(final_revenue_df):
    final_revenue_df['year_month_date'] = pd.to_datetime(final_revenue_df['year_month'], format='%Y %b')

    latest_month = final_revenue_df['year_month_date'].max()
    previous_month = (latest_month - pd.DateOffset(months=1))


    ## highest region
    top_region = final_revenue_df.groupby(['region'])['revenue'].sum().idxmax()
    top_region_rev = final_revenue_df[final_revenue_df.region == top_region]['revenue'].sum()
    filtered_data_region = final_revenue_df[final_revenue_df['region'] == top_region]
    latest_month_revenue_region = filtered_data_region[filtered_data_region['year_month_date'] == latest_month]['revenue'].sum()
    previous_month_revenue_region = filtered_data_region[filtered_data_region['year_month_date'] == previous_month]['revenue'].sum()

    # Calculate growth rate for region
    if previous_month_revenue_region == 0:
        growth_rate_region = 100.0
    else:
        growth_rate_region = ((latest_month_revenue_region - previous_month_revenue_region) / previous_month_revenue_region) * 100



    # highest country
    top_country = final_revenue_df.groupby(['country'])['revenue'].sum().idxmax()
    top_country_rev = final_revenue_df[final_revenue_df.country == top_country]['revenue'].sum()
    filtered_data_country = final_revenue_df[final_revenue_df['country'] == top_country]
    latest_month_revenue_country = filtered_data_country[filtered_data_country['year_month_date'] == latest_month]['revenue'].sum()
    previous_month_revenue_country = filtered_data_country[filtered_data_country['year_month_date'] == previous_month]['revenue'].sum()
    if previous_month_revenue_country == 0:
        growth_rate_country = 100.0
    else:
        growth_rate_country = ((latest_month_revenue_country - previous_month_revenue_country) / previous_month_revenue_country) * 100


    # total rev
    total_revenue = int(final_revenue_df['revenue'].sum())
    latest_month_revenue_total = final_revenue_df[final_revenue_df['year_month_date'] == latest_month]['revenue'].sum()
    previous_month_revenue_total = final_revenue_df[final_revenue_df['year_month_date'] == previous_month]['revenue'].sum()
    if previous_month_revenue_total == 0:
        growth_rate_total = 100.0
    else:
        growth_rate_total = ((latest_month_revenue_total - previous_month_revenue_total) / previous_month_revenue_total) * 100


    # highest prod
    top_prod = final_revenue_df.groupby(['product'])['revenue'].sum().idxmax()
    top_prod_rev = final_revenue_df[final_revenue_df['product'] == top_prod]['revenue'].sum()
    filtered_data_prod = final_revenue_df[final_revenue_df['product'] == top_prod]
    latest_month_revenue_prod = filtered_data_prod[filtered_data_prod['year_month_date'] == latest_month]['revenue'].sum()
    previous_month_revenue_prod = filtered_data_prod[filtered_data_prod['year_month_date'] == previous_month]['revenue'].sum()
    if previous_month_revenue_prod == 0:
        growth_rate_prod = 100.0
    else:
        growth_rate_prod = ((latest_month_revenue_prod - previous_month_revenue_prod) / previous_month_revenue_prod) * 100


    # avg monthly rev
    agg_revenue = final_revenue_df.groupby('year_month').revenue.sum().reset_index()
    agg_revenue['year_month_date'] = pd.to_datetime(agg_revenue['year_month'], format='%Y %b')
    agg_monthly_revenue = int(agg_revenue.revenue.mean())



    #CGMR
    delta = relativedelta(max(agg_revenue['year_month_date']), min(agg_revenue['year_month_date']))
    num_months = delta.years * 12 + delta.months + 1
    start_revenue = agg_revenue[agg_revenue['year_month_date'] == min(agg_revenue['year_month_date'])]['revenue'].values[0]
    end_revenue = agg_revenue[agg_revenue['year_month_date'] == max(agg_revenue['year_month_date'])]['revenue'].values[0]
    CMGR = ((end_revenue / start_revenue) ** (1/num_months)) - 1
    CMGR = CMGR * 100



    reg_sign = '+' if growth_rate_region > 0 else ''
    prod_sign = '+' if growth_rate_prod > 0 else ''
    count_sign = '+' if growth_rate_country > 0 else ''
    total_sign = '+' if growth_rate_total > 0 else ''
    compound_sign = '+' if CMGR > 0 else ''



    kpi_metrics = [
        {'title': 'Total Revenue', 'value': f"${total_revenue:,} <span style='color:{determine_color(growth_rate_total)}'>({total_sign}{int(growth_rate_total)}% P1M)</span>", 'unit': ''},
        {'title': 'Average Monthly Revenue', 'value': f"{agg_monthly_revenue:,}", 'unit': '$'},
        {'title': 'Compound Monthly Growth Revenue', 'value': f"<span style='color:{determine_color(CMGR)}'>{compound_sign}{round(CMGR,2)}%</span>", 'unit': ''},
        {'title': 'Highest Revenue Region', 'value': f"{top_region} — ${int(top_region_rev):,} <span style='color:{determine_color(growth_rate_region)}'>({reg_sign}{int(growth_rate_region)}% P1M)</span>", 'unit': ''},
        {'title': 'Highest Revenue Country', 'value': f"{top_country} — ${int(top_country_rev):,} <span style='color:{determine_color(growth_rate_country)}'>({count_sign}{int(growth_rate_country)}% P1M)</span> ", 'unit': ''},
        {'title': 'Highest Revenue Product', 'value': f"{top_prod} — ${int(top_prod_rev):,} <span style='color:{determine_color(growth_rate_prod)}'>({prod_sign}{int(growth_rate_prod)}% P1M)</span> ", 'unit': ''},
    ]

    # Create two rows for the KPIs
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)


    # Display the KPIs in two rows
    for i, metric in enumerate(kpi_metrics):
        if i < 3:
            col = [col1, col2, col3][i]
        elif i>=3:
            col = [col4, col5, col6][i - 3]



        with col:
            st.markdown(
                f'<div style="padding: 10px; margin: 10px; border-radius: 10px; '
                f'box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); border: 1px solid #ccc; '
                f'text-align: left;">'
                f'<h3 style="font-size: 18px;">{metric["title"]}</h3>'
                f'<p style="font-size: 24px; margin-bottom: 4px;">{metric["unit"]}{metric["value"]} </p></div>', 
                unsafe_allow_html=True
            )

def plot_product_region_bar(final_revenue_df):
    # Group by region and country for aggregated revenue
    grouped_data = final_revenue_df.groupby(['region', 'product'])['revenue'].sum().reset_index()

    # Calculate total revenue for each region
    region_totals = final_revenue_df.groupby('region')['revenue'].sum()

    color_map = {
        'premium': '#1F7368',
        'core': '#B185B4',
        'promos': '#A6C1D8',
        'custom_websites': '#FF7F0E',
        'basic_insights': '#9467BD',
        'adv_insights': '#2CA02C',
        'super_insights': '#17BECF'
    }


    fig = go.Figure()

    for product in grouped_data['product'].unique():
        product_data = grouped_data[grouped_data['product'] == product]
        text_labels = []
        for idx, row in product_data.iterrows():
            revenue_share = (row['revenue'] / region_totals[row['region']]) * 100
            text_labels.append(f"<b>{row['revenue']/1e6:.1f}M\n({revenue_share:.0f}%)<b>")

        fig.add_trace(go.Bar(
            x=product_data['region'],
            y=product_data['revenue'],
            name=product,
            marker_color=color_map.get(product),
            text=text_labels,
            textposition='inside',  # Place text inside the bars
            insidetextanchor='middle'
        ))

    fig.update_layout(
        barmode='stack',  # This makes the bar chart stacked
        height=800,
        width=800,
        yaxis_title="Revenue $USD",
        font=dict(size=15, color="#7f7f7f"),
        legend=dict(
            traceorder='normal',  # Sort legend entries alphabetically
        )
    )

    # Show the figure
    st.plotly_chart(fig)



def plot_country_region_bar(final_revenue_df):
    # Group by region and country for aggregated revenue
    grouped_data = final_revenue_df.groupby(['region', 'country'])['revenue'].sum().reset_index()

    # Calculate total revenue for each region
    region_totals = final_revenue_df.groupby('region')['revenue'].sum()

    country_color_map = {
        'Country 1': '#1F7368',
        'Country 2': '#B185B4',
        'Country 3': '#A6C1D8',
        'Country 4': '#FF7F0E',
        'Country 5': '#9467BD',
        'Country 6': '#2CA02C',
        'Country 7': '#17BECF',
        'Country 8': '#7F7F7F',
        'Country 9': '#D62728',
        'Country 10': '#8C564B',
        'Country 11': '#E377C2',
        'Country 12': '#7F7F7F',
        'Country 13': '#BCBD22',
        'Country 14': '#17BECF',
        'Country 15': '#1F77B4'
    }

    fig = go.Figure()

    for country in grouped_data['country'].unique():
        country_data = grouped_data[grouped_data['country'] == country]
        text_labels = []
        for idx, row in country_data.iterrows():
            revenue_share = (row['revenue'] / region_totals[row['region']]) * 100
            text_labels.append(f"<b>{row['revenue']/1e6:.1f}M\n({revenue_share:.0f}%)<b>")

        fig.add_trace(go.Bar(
            x=country_data['region'],
            y=country_data['revenue'],
            name=country,
            marker_color=country_color_map.get(country),
            text=text_labels,
            textposition='inside',  # Place text inside the bars
            insidetextanchor='middle'
        ))

    fig.update_layout(
        barmode='stack',  # This makes the bar chart stacked
        height=800,
        width=800,
        yaxis_title="Revenue $USD",
        font=dict(size=15, color="#7f7f7f"),
        legend=dict(
            traceorder='normal',  # Sort legend entries alphabetically
        )
    )

    # Show the figure
    st.plotly_chart(fig)



def get_dau_mau(daily, monthly):
    if daily == 0:
        return 0
    else:
        return daily / monthly * 100
    
def get_sticky_data(sticky_df):
    products = ['overall','premium' ,'core','promos','custom_websites','basic_insights','adv_insights','super_insights']
    for product in products:
        sticky_df[product + '_dau_mau'] = sticky_df.apply(lambda x: get_dau_mau(x[product + "_daily_active"], x[product + "_monthly_active"]), axis = 1)
    regional_dau_mau = sticky_df[['month_year_period','day_partition', 'region'] + [col for col in sticky_df.columns if 'dau_mau' in col]]
    regional_dau_mau = regional_dau_mau[['month_year_period','region','overall_dau_mau',
        'premium_dau_mau', 'core_dau_mau', 'promos_dau_mau',
        'custom_websites_dau_mau', 'basic_insights_dau_mau',
        'adv_insights_dau_mau', 'super_insights_dau_mau']].copy()
    regional_dau_mau_final = regional_dau_mau.groupby(['month_year_period','region']).agg('mean').reset_index()

    return regional_dau_mau_final



def plot_dau_mau_time_series(reg_dau_mau_final):

    regional_dau_mau_final = reg_dau_mau_final.copy()

    # Convert the month_year_period column to string format
    regional_dau_mau_final['month_year_period'] = regional_dau_mau_final['month_year_period'].astype(str)

    # Given data
    products = ['overall','premium', 'core', 'promos', 'custom_websites', 'basic_insights', 'adv_insights', 'super_insights']

    # Color map for the products
    color_map = {
        'premium': '#1F7368',
        'core': '#B185B4',
        'promos': '#A6C1D8',
        'custom_websites': '#FF7F0E',
        'basic_insights': '#9467BD',
        'adv_insights': '#2CA02C',
        'super_insights': '#17BECF',
        'overall': "black"
    }

    # Create a base layout with a title and axis labels
    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='DAU/MAU Ratio'),
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label=region,
                        method='update',
                        args=[{'visible': [region == r for r in regional_dau_mau_final['region'].unique()]},
                            {'title': f"{region}"}]) for region in regional_dau_mau_final['region'].unique()
                ])
            )

        ],
                height = 800,
            width = 1200
    )

    # Create a list to hold all our lines (traces)
    traces = []

    # Iterate over each region
    for region in regional_dau_mau_final['region'].unique():
        # Filter dataframe for the specific region
        regional_data = regional_dau_mau_final[regional_dau_mau_final['region'] == region]
        
        # For each product, create a trace (line)
        for product in products:
            traces.append(go.Scatter(
                x=regional_data['month_year_period'],
                y=regional_data[f"{product}_dau_mau"],
                mode='lines+markers',
                name=product,
                visible=(region == regional_dau_mau_final['region'].unique()[0]),
                line=dict(color=color_map[product])  # Specify the color from the color map
            ))

    # Create the figure and plot
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

def plot_dau_mau_region(reg_dau_mau_final):
    reg_dau_mau_final = reg_dau_mau_final.copy()
    
    products = ['overall','premium', 'core', 'promos', 'custom_websites', 'basic_insights', 'adv_insights', 'super_insights']
    growth_df = pd.DataFrame(columns=['region'] + [f"{product}_growth_rate" for product in products])

    for region in regional_dau_mau_final['region'].unique():
        latest_month = regional_dau_mau_final[regional_dau_mau_final.month_year_period == regional_dau_mau_final.month_year_period.max()]['month_year_period'].values[0]
        three_months_ago = latest_month - 3

        start_df = regional_dau_mau_final[(regional_dau_mau_final['month_year_period'] == three_months_ago) & (regional_dau_mau_final['region'] == region)]
        end_df = regional_dau_mau_final[(regional_dau_mau_final['month_year_period'] == latest_month) & (regional_dau_mau_final['region'] == region)]
        
        # Calculate growth rates
        growth_rates = [region]
        for product in products:
            start_value = start_df[f"{product}_dau_mau"].values[0]
            end_value = end_df[f"{product}_dau_mau"].values[0]
            
            growth_rate = ((end_value - start_value) / start_value) * 100
            growth_rates.append(growth_rate)
            
        growth_df.loc[len(growth_df)] = growth_rates

    latest_dau_mau = regional_dau_mau_final[(regional_dau_mau_final['month_year_period'] == latest_month)]

    melted_june_data = pd.melt(latest_dau_mau, id_vars='region', value_vars=[f"{product}_dau_mau" for product in products], var_name='product', value_name='dau/mau')
    melted_june_data['product'] = melted_june_data['product'].str.replace('_dau_mau', '')
    melted_growth_df = pd.melt(growth_df, id_vars='region', value_vars=[f"{product}_growth_rate" for product in products], var_name='product', value_name='growth')
    melted_growth_df['product'] = melted_growth_df['product'].str.replace('_growth_rate', '')
    final_growth_df = melted_june_data.merge(melted_growth_df, on=['region', 'product'])

    # Color map
    color_map = {
        'Region 1': '#1F7368',
        'Region 2': '#B185B4',
        'Region 3': '#A6C1D8'
    }

    # Calculate average dau/mau per product
    avg_dau_mau = final_growth_df.groupby('product')['dau/mau'].mean().sort_values(ascending=False)

    # Extracting sorted products for plotting
    sorted_products = avg_dau_mau.index.tolist()
    regions = final_growth_df['region'].unique()

    # Create figure
    fig = go.Figure()

    annotations = []

    # The width of each bar when using barmode='group' in plotly
    bar_width = 0.8 / len(regions)

    # Add bars for each region
    for idx, region in enumerate(regions):
        region_data = final_growth_df[final_growth_df['region'] == region]
        region_data = region_data.set_index('product').loc[sorted_products].reset_index()

        fig.add_trace(
            go.Bar(
                x=region_data['product'],
                y=region_data['dau/mau'],
                name=region,
                marker_color=color_map[region],
                text=region_data['dau/mau'].round(1).astype(int).astype(str) + '%',  # text for inside bar
                textposition='auto',
                textfont=dict(size = 14)
            )
        )

        # Add growth labels as annotations
        for i, (product, dau_mau, growth) in enumerate(region_data[['product', 'dau/mau', 'growth']].values):
            if math.isnan(growth):
                growth = 0.0
                
            growth_color = 'green' if growth >= 0 else 'red'
            growth_sign = '+' if growth > 0 else ''
            offset = -0.4 + idx * bar_width + 0.5 * bar_width  # calculate position offset for grouped bars
            annotations.append(
                dict(
                    x=i + offset,  # adjust x position using the offset
                    y=dau_mau + 2,  # slight offset to position text above bar
                    xref="x",
                    yref="y",
                    text=f"{growth_sign}{growth:.0f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color=growth_color
                    )
                )
            )

    fig.update_layout(
        barmode='group',
        xaxis_title="Product",
        yaxis_title="DAU/MAU Ratio",
        height=800,
        width=1500,
        annotations=annotations,
            legend=dict(
            x=0.9,
            y=0.89
        ),
    )



    

    st.plotly_chart(fig)
    
    # Sort final_growth_df by the 'growth' column 
    final_growth_df = final_growth_df.sort_values(by="growth", ascending=False)
    final_growth_df.columns  = ['Region','Product','DAU/MAU Ratio (%)', 'P3M Growth (%)']
    final_growth_df['DAU/MAU Ratio (%)'] = final_growth_df['DAU/MAU Ratio (%)'].round(1).astype(str)
    final_growth_df['P3M Growth (%)'] = final_growth_df['P3M Growth (%)'].round(1).astype(str)

    st.subheader("Trending Sticky Products ")
    # Apply a gradient background for 'growth' and 'dau/mau' columns
    styled_df = final_growth_df.style.background_gradient(cmap='RdYlGn', subset=['P3M Growth (%)'])
    

    st.table(styled_df)


def get_adoption_table(df):
    new_users = df.copy()

    # Determine the last day of each month for each region and country
    new_users['year_month'] = new_users['day_partition'].dt.to_period('M')
    last_day_per_month = new_users.groupby(['year_month', 'region', 'country']).day_partition.max().reset_index()
    
    # Filter the dataset to only include these last days
    filtered_df = pd.merge(new_users, last_day_per_month, on=['year_month', 'region', 'country', 'day_partition'])
    
    products = ['overall','premium', 'core', 'promos', 'custom_websites', 'basic_insights', 'adv_insights', 'super_insights']

    # Compute the new users gained for each product by taking the difference of the monthly_access value
    # from the current month's end to the last value of the previous month
    for product in products:
        if product != "overall":
            access_column = f"{product}_monthly_access"
            new_users_column = f"new_{product}_users"
            filtered_df[new_users_column] = filtered_df.groupby(['region', 'country'])[access_column].diff().fillna(0)
    
    # Reshape the DataFrame to get the desired output structure
    melted_df = filtered_df.melt(id_vars=['year_month', 'region', 'country'], 
                                 value_vars=[f"new_{product}_users" for product in products if product != "overall"], 
                                 var_name='product', 
                                 value_name='new users gained')
    
    # Rename the columns for clarity and filter out any records with zero new users
    melted_df['product'] = melted_df['product'].str.replace('new_', '').str.replace('_users', '')
    new_users_final_df = melted_df[melted_df['new users gained'] != 0]
    
    return new_users_final_df


def plot_new_users_gained(new_users_final_df):
    # Convert year_month to string
    new_users_final_df['year_month'] = new_users_final_df['year_month'].astype(str)
    
    # Sort the DataFrame by 'year_month'
    new_users_final_df = new_users_final_df.sort_values(by='year_month')
    
    # Group data
    grouped_data = new_users_final_df.groupby(['year_month', 'product'])['new users gained'].sum().reset_index()

    color_map = {
        'premium': '#1F7368',
        'core': '#B185B4',
        'promos': '#A6C1D8',
        'custom_websites': '#FF7F0E',
        'basic_insights': '#9467BD',
        'adv_insights': '#2CA02C',
        'super_insights': '#17BECF'
    }

    products = grouped_data['product'].unique()

    fig = go.Figure()
    
    # Loop through each product and add a bar
    for product in products:
        subset = grouped_data[grouped_data['product'] == product]
        fig.add_trace(
            go.Bar(
                x=subset['year_month'],
                y=subset['new users gained'],
                name=product,
                marker_color=color_map.get(product, 'grey')
            )
        )
    
    # Calculate monthly totals
    monthly_totals = grouped_data.groupby('year_month')['new users gained'].sum()

    # Calculate an offset for annotations based on the max bar height
    offset = 0.03 * np.max(np.abs(monthly_totals))
    
    # Add annotations
    annotations = []
    for month, total in monthly_totals.items():
        sign = "+" if total > 0 else ""
        position = total + offset if total > 0 else total - offset
        annotations.append(
            dict(
                x=month,
                y=position,
                xref="x",
                yref="y",
                text=f"<b>{sign}{int(total)}<b>",
                showarrow=False
            )
        )
    
    fig.update_layout(
        barmode='stack',
        xaxis_title='Date',
        yaxis_title='Total New Users Gained',
        bargap=0.1,
        annotations=annotations,
        height=800,
        width = 1500
    )
    
    st.plotly_chart(fig)



## Loading and Filtering the data ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = load_data()
min_date = df['day_partition'].min().to_pydatetime()
max_date = df['day_partition'].max().to_pydatetime()


# Sidebar section
st.sidebar.header('Data Filters')

# Multiple select dropdown for region selection
regions = ['ALL'] + list(df.region.unique())

selected_regions = st.sidebar.multiselect('Select Regions', regions, default=['ALL'])


# Slider for time range selection
start_date = st.sidebar.date_input('Start Date', min_date)
end_date = st.sidebar.date_input('End Date', max_date)

# Convert start_date and end_date to datetime.datetime objects
start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time()) + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)



# Filter the raw dataset based on selected regions and time range
if 'ALL' in selected_regions:
    filtered_data = df[(df['day_partition'] >= start_date) & (df['day_partition'] <= end_date)]
else:
    filtered_data = df[(df['region'].isin(selected_regions)) & (df['day_partition'] >= start_date) & (df['day_partition'] <= end_date)]






## Building the dashboard ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
logo_image_url = './logo_2.png'  # Replace with the actual URL of your logo image
st.image(logo_image_url, width=200)

st.title('Product Analytics Dashboard')
st.divider()


st.warning('Use the data filters sidebar to adjust the time range. The populated values are totals from the user set time range. Values given in $USD')

# Revenue section
st.header('Revenue')



final_revenue_df = get_revenueue_table(filtered_data)
plot_revenue_kpis(final_revenue_df)
st.subheader("")
st.subheader('Monthly Revenue by Region')
plot_total_revenue(final_revenue_df)
st.subheader('Monthly Revenue by Product')
plot_product_revenue(final_revenue_df)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Revenue Share by Country')
    plot_country_region_bar(final_revenue_df)
with col2:
    st.subheader('Revenue Share by Product')
    plot_product_region_bar(final_revenue_df)




st.divider()


st.header('Product Stickiness')

st.subheader("AVG. DAU/MAU Ratio by Region")
st.warning("DAU/MAU Ratio is calculated by averaging daily dau/mau by month. The latest month is used for the charts below. P3M Growth is given above each regions' bar")
regional_dau_mau_final = get_sticky_data(filtered_data)
plot_dau_mau_region(regional_dau_mau_final)



st.subheader('AVG. DAU/MAU Ratio Time Series')
st.warning("DAU/MAU Ratio is calculated by averaging daily DAU/MAU by month. The latest month is used for the charts below. Use the Region filter inside the chart below to change regions.")

plot_dau_mau_time_series(regional_dau_mau_final)



st.divider()

# Product Adoption section
st.header('Product Adoption')
st.subheader("Net User Gain/Loss")
new_users_final_df = get_adoption_table(filtered_data)
plot_new_users_gained(new_users_final_df)






