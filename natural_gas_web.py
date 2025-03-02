import streamlit as st
import plotly.express as px
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import os

st.set_page_config(
    page_title = 'Global natural gas production map',
    layout = "wide")


## helper function
def get_df(natural_gas_df, cumu_df):
    df = pd.melt(natural_gas_df, id_vars='Year', var_name='Country', value_name='Value')
    df_dropna = df.dropna()
    non_numeric_indices = df_dropna[pd.to_numeric(df_dropna['Value'], errors='coerce').isna()].reset_index()
    for i in range(len(non_numeric_indices)):
        idx = non_numeric_indices.loc[i, 'index']
        val = df.loc[idx, 'Value']
        new_val = val.year*0.0001 + val.month
        print(f'before:{df.loc[idx, 'Value']}')
        df.loc[idx, 'Value'] = new_val
        print(f'after:{df.loc[idx, 'Value']}\n')
    cumu_df.loc[len(cumu_df)] = ['World', cumu_df['Cumulative production'].sum(axis=0)]
    cumu_df_1 = cumu_df.copy()
    cumu_df_1['Year'] = '1900-2022'
    cumu_df_1.rename(columns={'Cumulative production':'Value'}, inplace=True)
    df = pd.concat([df, cumu_df_1])
    df.replace({'Tanzania':'United Republic of Tanzania', 'The Bahamas':'Bahamas', 'United States': 'United States of America', 'Eswatini': 'eSwatini'}, inplace=True)
    return df

def get_year_df(df, year):
    """
    Returns a DataFrame containing data for the specified year.

    Parameters:
        year (str or int): year range from 'year_list', including options like 1900 and '1900-2022'.

    Note:
        - For 1900 only year production data and percentage data are returned.
        - For '1900-2022' only cumulative production data and percentage data are returned.
        - For other years, year production, percentage and increasing rate data are provided.
    """

    def add_percentage(df, col='Value'):
        a = df[df['Country']=='World'][col].values[0]
        df['Percentage'] = round(df[col]/a * 100, 2)
        return df

    def add_increasing_rate(df, year):
        df_1 = df[df['Year']==year-1]
        df_2 = df[df['Year']==year]
        name_1 = 'Value_'+str(year-1)
        name_2 = 'Value_'+str(year)
        df_merged = pd.merge(df_1[['Country', 'Value']], df_2[['Country', 'Value']],
                      on='Country', suffixes=(f'_{year-1}', f'_{year}'), how='right')
        df_merged.dropna(subset=name_2, inplace=True)

        df_merged[name_1] = pd.to_numeric(df_merged[name_1], errors='coerce')
        df_merged[name_2] = pd.to_numeric(df_merged[name_2], errors='coerce')
        df_merged.loc[~df_merged[name_1].isna() & (df_merged[name_1] != 0), 'Increasing rate'] = round((df_merged[name_2] - df_merged[name_1]) / df_merged[name_1] * 100, 2)
        df_merged.loc[(df_merged[name_1] == 0) | df_merged[name_1].isna(), 'Increasing rate'] = np.nan

        df_merged.drop(columns=[name_1], inplace=True)
        df_merged.rename(columns={name_2:'Value'}, inplace=True)

        return df_merged

    if year =='1900-2022':
        df_y = df[df['Year']==year]
        df_y.drop(columns=['Year'], inplace=True)
        df_y['Value'] = pd.to_numeric(df_y['Value'], errors='coerce')
        df_y = add_percentage(df_y)

    elif year == 1900:
        # no increasing rate
        df_y = df[df['Year']==year]
        df_y.drop(columns=['Year'], inplace=True)
        df_y['Value'] = pd.to_numeric(df_y['Value'], errors='coerce')
        df_y = add_percentage(df_y)

    else:
        df_y = add_increasing_rate(df, year)
        df_y = add_percentage(df_y)
        
    df_y['Value'] = round(df_y['Value'],2)

    return df_y

def merge_geo(world, df_y):
    df_y_map = world.merge(df_y, how='left', left_on='SOVEREIGNT', right_on='Country')
    return df_y_map

def create_gif(df, world):
    # reference: https://youtu.be/Wyp1fH9txsE?si=MCYHygPr9tUHz63_
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgrey')  # Plot base map
    df_1 = df[(df['Year']!='1900-2022') & (df['Country']!='World')]
    years = sorted(df_1['Year'].unique())
    df_1['Value'] = pd.to_numeric(df_1['Value'], errors='coerce')
    df_1 = df_1.dropna(subset=['Value']) 
    vmin, vmax = 0, df_1['Value'].max()
    # print(f'vmax: {vmax}')
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.3)
    cbar.set_label("Year production (EJ)", fontsize=8)
    cbar.ax.set_position([0.15, 0.4, 0.2, 0.1]) 
    
    # Define the custom color scale
    custom_colors = [
        [0, '#F2E1A3'],
        [0.033, '#F2D07A'],
        [0.067, '#F1BD56'],
        [0.1, '#F0A637'],
        [0.134, '#F07F1D'],
        [0.34, '#E86D14'],
        [0.48, '#E35710'],
        [0.68, '#D83D0D'],
        [1, '#D12608']
    ]
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', [color[1] for color in custom_colors], N=256)
    original_cmap = plt.cm.OrRd
    colors = original_cmap(np.linspace(0.1, 1, 256))  # Skip the first 10% (white)
    new_cmap = mcolors.LinearSegmentedColormap.from_list('OrRd_no_white', colors)

    def animate(idx):
        ax.clear()  
        world.plot(ax=ax, color='lightgrey') 
        
        year = years[idx]
        # print(f'idx: {idx}; year: {year}')
        df_year = df[df['Year'] == year]
        # print(f'got df_year for {year}!')
        df_world = world.merge(df_year, left_on='SOVEREIGNT', right_on='Country', how='left')
        
        df_world.plot(column='Value', ax=ax, legend=False, vmin=vmin, vmax=vmax, 
                    #   cmap=cmap
                    # cmap = 'OrRd'
                    cmap = new_cmap
                      )
        
        print(f'Finish df_world plot for year {year}!')
        
        ax.set_title(f"Year: {year}", fontsize=16)
        ax.axis('off') 

    ani = FuncAnimation(fig, animate, frames=len(years), interval=5) 
    
    # Save animation as a GIF
    gif_path = "./data/gif/animated_map.gif"
    output_dir = os.path.dirname(gif_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ani.save(gif_path, writer='pillow', fps=2)
    
    return gif_path

def extract_yearly_data(df_y):
    world_production = df_y.loc[df_y['Country'] == 'World', 'Value'].values[0]
    df_y_c = df_y[df_y['Country']!='World']
    max_value = df_y_c['Value'].max()
    max_country = df_y_c.loc[df_y_c['Value'].idxmax(), 'Country']
    percentage = round((max_value/world_production)*100, 2)
    return max_country, max_value, percentage

def extract_cumulative_data(cumu_df):
    cumu_df.rename(columns = {'Cumulative production':'Value'}, inplace = True)
    world_production = cumu_df[cumu_df['Country']=='World']['Value'].iloc[0]
    cumu_df_c = cumu_df[cumu_df['Country']!='World']
    cumu_df_sorted = cumu_df_c.sort_values(by='Value', ascending = False).reset_index()
    country_1, value_1 = (cumu_df_sorted.loc[0, 'Country'], cumu_df_sorted.loc[0, 'Value'])
    country_2, value_2 = (cumu_df_sorted.loc[1, 'Country'], cumu_df_sorted.loc[1, 'Value'])
    
    return world_production, country_1, value_1, country_2, value_2


def draw_color_map(df_y_map, year):
    # reference: https://youtu.be/qY9IIH8Tsgg?si=OMDs7CPxvI9jyLCU
    custom_data_list = ['Percentage']
    if year != 1900 and not isinstance(year, str):
        custom_data_list.append('Increasing rate')
    
    # improvement: change the color palatte for single year (different data scale)
    color_continuous_scale = [[0, '#D9D689'], [0.033, '#D9B459'],[0.067, '#D99854'], [0.1, '#D47E4F'], [0.134, '#CB6445'], 
                                [0.34, '#BB4A46'], [0.48, '#A93647'], [0.68, '#8B2F48'], [1, '#622F4E']]
    if year != '1900-2022':
        color_continuous_scale=[
            [0, '#F2E1A3'],
            [0.033, '#F2D07A'],
            [0.067, '#F1BD56'],
            [0.1, '#F0A637'],
            [0.134, '#F07F1D'],
            [0.34, '#E86D14'],
            [0.48, '#E35710'],
            [0.68, '#D83D0D'],
            [1, '#D12608']
        ]
        
    
    fig = px.choropleth(data_frame = df_y_map,
        geojson=world, 
        locations='Country',    
        color="Value",
        featureidkey = "properties.SOVEREIGNT",
        hover_name="Country",  
        hover_data = {"Country": False},
        custom_data= custom_data_list,
        projection="natural earth",    
        color_continuous_scale=color_continuous_scale  
        )
    
    fig.update_geos(
        visible=False,  
        showcountries=True,  
        countrycolor="gray",  
        fitbounds="locations"
    )
    fig.update_layout(
        margin={"r": 0, "l": 0}, # improvement: adjust margin to make the empty space less
        coloraxis_colorbar=dict(
            title="<b>Production amount (EJ)</b>" if year!='1900-2022' else "<b>Cumulative production (EJ)</b>",  # Bold title
            titlefont=dict(size=12),  # Title font customization
            tickfont=dict(size=9.5),  # Customize font size of the tick labels
            orientation="h",
            len=0.35,  # Adjust length of the color bar
            thickness=15,
            x=0.2,  # Position color bar near top left
            y=0.92,
        ),
        title={
            'text': f"<b>Natural gas production by country, {year}</b>",
            'font': {'size': 18, 'color': 'black'},  
            'x': 0.01,  # Position title near top left
        }
    )
    hovertemplate = ("<b>%{location}</b><br>" +  # Country name in bold
                            "Production: %{z} EJ<br>"
                            + "Percentage of the World production: %{customdata[0]}%")
    if year != 1900 and not isinstance(year, str):
        hovertemplate =  ("<b>%{location}</b><br>" +  # Country name in bold
                            "Production: %{z} EJ<br>"
                            + "Percentage of the World production: %{customdata[0]}%<br>"
                            + 'Increasing rate from the last year: %{customdata[1]}%')
    fig.update_traces(
        hovertemplate= hovertemplate,
        hoverlabel = dict(bgcolor = 'white')
    )
    # improvement: some coutries have value 0, so should colorbar start with 0
    if year == '1900-2022':
        fig.update_layout(coloraxis_cmin=0, coloraxis_cmax=1468)
        
    return fig        


if __name__ == "__main__":
    natural_gas_df = pd.read_excel('./data/The history of global natural gas production.xlsx', sheet_name = 'data line chart')
    cumu_df = pd.read_excel('./data/The history of global natural gas production.xlsx', sheet_name = 'data for map')
    world = gpd.read_file("./data/countries/ne_110m_admin_0_countries.shp") # change it into your file path
    
    df = get_df(natural_gas_df, cumu_df)
    gif_path = create_gif(df, world)
    
    menu = ['Cumulative production', 'Annual production', 'Production trends']
    choice = st.sidebar.selectbox('Select View for Natural Gas Production', menu)
    if choice =='Cumulative production':
        st.title('Cumulative Natural Gas Production Map')
        # st.header('Cumulative production')
        df_cumu = get_year_df(df, '1900-2022')
        world_production, country_1, value_1, country_2, value_2 = extract_cumulative_data(cumu_df)
        st.write(f'The cumulative production of world from 1900 to 2022 is {world_production} EJ.')
        st.write(f'{country_1} has the highest cumulative natural gas production at {value_1} EJ, followed by {country_2} with {value_2} EJ.')
        fig = draw_color_map(df_cumu, '1900-2022')
        st.plotly_chart(fig, use_container_width=True)
        
    elif choice == 'Annual production':
        st.title("Annual Natural Gas Production")
        st.header("Production Amount and Growth Rate from Previous Year")
        year = st.slider('Select a year to display', 1900, 2022, step=1, value=2022)
        df_y = get_year_df(df, year)
        
        max_country, max_value, percentage = extract_yearly_data(df_y)
        st.write('\n')
        st.write(f"In {year}, {max_country} produced the most natural gas of {max_value:.3f} EJ, accounting for {percentage:.2f}% of the world production during that year.")
        st.write(f"In today's scale, {max_value:.3f} EJ can supply the entire world's electricity consumption for about {max_value/1.7:.1f} days.")
        
        df_y_map = merge_geo(world, df_y)
        fig = draw_color_map(df_y_map, year)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.title('Global Natural Gas Production Trend')
        st.write('Natural gas production globally saw a significant increase starting in the 1930s, with many countries beginning to tap into their natural gas resources.')
        st.write('By the 1950s, production experienced a sharp jump due to advancements in extraction technologies and rising energy demands. \nThe 1970s marked another surge in production, driven by the oil crises and increased focus on alternative energy sources.')
        st.write('From the 1980s onwards, production continued to grow steadily, with major producers expanding their capacity and emerging markets starting to contribute more significantly to global output.\n')
        st.header('History of natural gas production, 1900-2022')
        # gif_path = "../data/animated_map.gif"
        st.image(gif_path, use_column_width=True)



