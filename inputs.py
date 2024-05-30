import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf


#filename = './FRA_AR_Grenoble.074850_TMYx.epw'
filename = './FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2007-2021.epw'


[data, meta] = read_epw(filename, coerce_year=None)
data

# Extract the month and year from the DataFrame index with the format 'MM-YYYY'
month_year = data.index.strftime('%m-%Y')

# Create a set of unique month-year combinations
unique_month_years = sorted(set(month_year))

# Create a DataFrame from the unique month-year combinations
pd.DataFrame(unique_month_years, columns=['Month-Year'])

# select columns of interest
weather_data = data[["temp_air", "dir_n_rad", "dif_h_rad",]]

# replace year with 2023 in the index 
weather_data.index = weather_data.index.map(
    lambda t: t.replace(year=2023))

#print(weather_data.loc['2000-06-29 12:00'])

# Define start and end dates
start_date = '2023-01-01 00:00'
end_date = '2023-12-31 23:00'         # time is 00:00 if not indicated

# Filter the data based on the start and end dates
donne = weather_data.loc[start_date:end_date]
print (donne)

######### TEMPERATURES SOURCES ##########

weather = weather_data.loc[start_date:end_date]
To = weather['temp_air']

# indoor air temperature set-point
Ti_sp = pd.Series(20, index=To.index)

Ti_day, Ti_night = 20, 16

Ti_sp = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

######### HEAT SOURCES ##########

## Orientation du mur ##
surface_orientation = {'slope': 90,     # 90° is vertical; > 90° downward
                       'azimuth': 45,    # 0° South, positive westward
                       'latitude': 45.19}  # °, North Pole 90° positive

albedo = 0.3

rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientation, albedo)

Etot = rad_surf.sum(axis=1)

# window glass properties
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

# Outdoor wall surface
α1 = 0.25       # short wave absorbtivity indoor white walls
α2 = 0.30       # short wave absorbtivity indoor walls

# solar radiation absorbed by the outdoor surface of the wall
Φo = α2 * Sc * Etot

# solar radiation absorbed by the indoor surface of the wall
Φi = τ_gSW * α1 * Sw*2 * Etot

# auxiliary (internal) sources
Qa = 0 * np.ones(weather.shape[0])

# Input data set
input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp,
                               'Φo': Φo, 'Φi': Φi, 'Qa': Qa,
                               'Etot': Etot})

