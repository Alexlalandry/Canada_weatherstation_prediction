import streamlit as st
import numpy as np
import pandas as pd
import dill
import datetime
from datetime import date
import zipfile
import folium
from streamlit_folium import st_folium
import plotly.express as px

#@st.cache_data

# Import models
def load_modelmax():
    with zipfile.ZipFile("model_maxtemp_dill_minimal.zip", "r") as zip_ref:
        with zip_ref.open("model_maxtemp_dill_minimal.dill", "r") as file:
            data = dill.load(file)
    return data

def load_minmodel():
    with zipfile.ZipFile("model_mintemp_dill_minimal.zip", "r") as zip_ref:
        with zip_ref.open("model_mintemp_dill_minimal.dill", "r") as file:
            data = dill.load(file)
    return data

def load_meanmodel():
    with zipfile.ZipFile("model_meantemp_dill_minimal.zip", "r") as zip_ref:
        with zip_ref.open("model_meantemp_dill_minimal.dill", "r") as file:
            data = dill.load(file)
    return data

data_max = load_modelmax()
regressor_max = data_max["model"]
data_min = load_minmodel()
regressor_min = data_min["model"]
data_mean = load_meanmodel()
regressor_mean = data_mean["model"]

encoder = data_max["encoder"]
scaler = data_max["scaler"]

categorical = ['Province','Season']
ordinal = ['Month','Year','Day', 'Season_num']
continuous = ['Longitude (x)','Latitude (y)','Elevation (m)']
numerical = continuous+ordinal
drop = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)','Total Precip (mm)', 'Station Name']
columns = {'Categorical':categorical,
                'Ordinal':ordinal, 
                'Continuous':continuous,
                'Numerical':numerical,
                'drop':drop}

# Function to map the season in the dataframe:
def map_season_num(month):
    season_num = {1:[12,1,2], 2:[3,4,5], 3:[6,7,8], 4:[9,10,11]}
    for key, value in season_num.items():
        if month in value:
            return key
        
def map_season(month):
    season1 = {'Winter':[12,1,2], 'Spring':[3,4,5], 'Summer':[6,7,8], 'Fall':[9,10,11]}
    for key, value in season1.items():
        if month in value:
            return key

# application function
def show_predict_page():

    st.title('Weater stations in Canada, historical temperature predictions')

    st.write("""### Please select the province""")
    station = pd.read_csv('Station_info.csv')

    provinces = ('ALBERTA', 
                'BRITISH COLUMBIA', 
                'QUEBEC',
                'ONTARIO',
                'SASKATCHEWAN', 
                'MANITOBA', 
                'NOVA SCOTIA', 
                'NUNAVUT', 
                'NEWFOUNDLAND', 
                'NORTHWEST TERRITORIES', 
                'NEW BRUNSWICK', 
                'YUKON TERRITORY', 
                'PRINCE EDWARD ISLAND',)
    
    metrics = ['Max Temp (°C)',
               'Min Temp (°C)',
               'Mean Temp (°C)']
    # Date limit from models training
    ulti_min = date(2023,1,1)
    ulti_max = date(2023,12,31)
   
    # Province selection for the map zoom
    selected_province = st.selectbox("Select a province", provinces)
    # Date selections for data prediction
    min_date = st.date_input(f'Select a start date, min date {ulti_min}',
                              value=None,
                              min_value=ulti_min,
                              max_value=ulti_max)
    max_date = st.date_input(f'Select an end date, max date {ulti_max}',
                              value=None,
                              min_value=ulti_min,
                              max_value=ulti_max)
    
    prov_station = station.loc[station['Province']==selected_province]
    prov_station = prov_station.rename({'Latitude (y)':'latitude',
                                        'Longitude (x)':'longitude'}, axis=1)
    
    # Province center coordinate for the map location
    province_center = {'ALBERTA': (55.000000, -115.000000)
                    ,'BRITISH COLUMBIA': (53.726669, -127.647621)
                    ,'QUEBEC':(53.000000, -70.000000)
                    ,'ONTARIO':(50.000000, -85.000000)
                    ,'SASKATCHEWAN':(55.000000, -106.000000) 
                    ,'MANITOBA':(56.415211, -98.739075)
                    ,'NOVA SCOTIA':(45.000000, -63.000000)
                    ,'NUNAVUT':(70.453262, -86.798981)
                    ,'NEWFOUNDLAND':(53.135509, -57.660435)
                    ,'NORTHWEST TERRITORIES':(64.2666656, -119.1833326)
                    ,'NEW BRUNSWICK':(46.498390, -66.159668)
                    ,'YUKON TERRITORY':(64.000000, -135.000000)
                    ,'PRINCE EDWARD ISLAND':(46.250000, -63.000000)
                       }

    # Creating the map
    st.header(f'Weather stations localisation for {selected_province}')
    map = folium.Map(location=province_center[selected_province], zoom_start=6)

    # Adding the stations coordinates on the map
    for i in range(len(prov_station)):
        location = prov_station.iloc[i]['latitude'], prov_station.iloc[i]['longitude']
        folium.Marker(location,
                      popup=prov_station.iloc[i]['StationId'],
                      tooltip=prov_station.iloc[i]['Station Name']).add_to(map)

    # Storing station selection made by the user
    output = st_folium(map, width=700, returned_objects=["last_object_clicked_popup"])

    selected_station_list = st.selectbox("Select a station", 
                                     prov_station['StationId'], 
                                     index=None,
                                     placeholder='Choose station in map or in list')

    # Creating function for making prediction
    def make_prediction(selected_station):
            station_name = prov_station['Station Name'].loc[prov_station['StationId'] == int(selected_station)].values[0]
            st.write('Start date, End date', min_date, max_date)
            longitude = prov_station['longitude'].loc[prov_station['StationId']==int(selected_station)].values[0]
            latitude = station['Latitude (y)'].loc[station['StationId']==int(selected_station)].values[0]
            elevation = station['Elevation (m)'].loc[station['StationId']==int(selected_station)].values[0]
           
            # Creating the dataset
            date_range = pd.date_range(start=min_date, end=max_date, freq='D').to_list()
            len_date = len(date_range)
            data = pd.DataFrame(data={'Date':date_range})
            data['Province'] = selected_province
            data['StationId'] = selected_station
            data['Longitude (x)'] = longitude
            data['Latitude (y)'] = latitude
            data['Year'] = pd.DatetimeIndex(data['Date']).year
            data['Month'] = pd.DatetimeIndex(data['Date']).month
            data['Season'] = data['Month'].apply(map_season)
            data['Season_num'] = data['Month'].apply(map_season_num)
            data['Day'] = pd.DatetimeIndex(data['Date']).day
            data['Elevation (m)'] = elevation

            # Preprocessing (encoder and scaler)
            data1 = data[['Longitude (x)','Latitude (y)', 'Year', 'Season_num', 'Month', 'Day',
                            'Elevation (m)', 'Province','StationId','Season','Date']]
            data_enc = encoder.transform(data1[columns['Categorical']])
            data2 = data1.drop(columns['Categorical'], axis=1)
            columns_encode = encoder.get_feature_names_out()
            data_enc2 = pd.DataFrame(data=data_enc, columns=columns_encode)
            data3 = pd.concat([data2,data_enc2], axis=1)
            data3[columns['Numerical']]= scaler.transform(data3[columns['Numerical']])
            X=data3.drop(['StationId','Date'],axis=1)

            # Weather predictions
            data['Max Temp (°C)'] = regressor_max.predict(X)
            data['Mean Temp (°C)'] = regressor_mean.predict(X)
            data['Min Temp (°C)'] = regressor_min.predict(X)

            

            # CMake data exportable
            csv = data.to_csv(index=False)
            st.subheader('Data extraction for selected station')
            tab1, tab2 = st.tabs(['Temperature estimation per date','Dataset for download'])
            with tab1:
                fig = px.line(data, x='Date', y=metrics,
                title=f'Weather prediction for station {selected_station}, {station_name} in {selected_province}')
                st.plotly_chart(fig,theme='streamlit')
            with tab2:
                st.write('Showing only the first 31 days, download the csv for full dataset')
                st.dataframe(data.head(31))
                st.download_button(
                    "Press to Download csv",
                    csv,
                    f"{selected_province}_{selected_station}_{min_date}_{max_date}.csv",
                    "text/csv",
                    key='download-csv')


    if min_date and max_date:
        if selected_station_list:
            st.write('Station selected:', selected_station_list)
            make_prediction(selected_station_list)
        elif output["last_object_clicked_popup"]:
            st.write('Station selected:', output["last_object_clicked_popup"])
            make_prediction(output["last_object_clicked_popup"]) # adding elif for station list box
           
        else:
            st.write('Please select a station in the map or in the list')
    else:
        st.write('Please select the start date and end date')

## Next: date and int format correction
## adding error on metrics
### adding a map with the pourcentage of missing values per station for all station??
## adding also station with more than 30% missing values???

