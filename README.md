## Project title
### Wild fire risk prediction for Canada

## Project description: 
The first idea behind this project was to predict wildfire risk base on region and on past season weather.
Weather data came from [Gouvernment of Canada](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html) where weather station data is accessible.
The problem is that there is a lot of missing values and some stations in remote area don't log data anymore.

### Phase 1:
The present application predict temperature min, max and mean per day for weather station around all Canada. With the dataset resulting from the application, you can replace dthe missing data from the station that you are interested in.

To do the prediction, I have trained 3 Linear models with scikit-learn DecisionTreeRegressor. The features are the coordinates of each station, the elevation, the date and the season. I have also tested different linear model. Overall, the RandomForestRegressor() had the most accurate results but it is very heavy to trained and to deploy. For this reason, I have decide to use the DecisionTreeRegressor. 


### Phase 2: For the future
I would like to add a model to predict the precipitation received. The current model in [Wildfire_preprocess_model](_Wildfire_preprocess_model.ipynb) for precicipitation has a R2 score of 50%. This model is in the notebook but not deploy.
I would also like to improve the weather temperature prediction by adding more feature. Finding historical data on atmospheric pressure, air humudity and wind for example can be interesting for predicting precipitation and temperature.
Finally I would also like to train an other set of models for past 2020 data.

For the streamlit app., I would like to add the option for selecting more than 1 station ID per extraction.

### Phase 3: adding the wildfire risk prediction
I have found data on historical wildfire in Canada, but I did not have coordinates per fire, only a summation per province and per year from 1990 to 2020. I have to do more research to find interesting data
