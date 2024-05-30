import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf

#filename = './weather_data/FRA_AR_Grenoble.074850_TMYx.epw'
filename = './weather_data/FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2007-2021.epw'
# filename = './weather_data/FRA_AR_Lyon-Bron.AP.074800_TMYx.2004-2018.epw'


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

# replace year with 2000 in the index 
weather_data.index = weather_data.index.map(
    lambda t: t.replace(year=2024))

#print(weather_data.loc['2000-06-29 12:00'])

# Define start and end dates
start_date = '2024-04-29 00:00'
end_date = '2024-05-01 00:00'         # time is 00:00 if not indicated

# Filter the data based on the start and end dates
donne = weather_data.loc[start_date:end_date]
print (donne)


## Orientation du mur ##
surface_orientation = {'slope': 90,     # 90° is vertical; > 90° downward
                       'azimuth': 0,    # 0° South, positive westward
                       'latitude': 45.19}  # °, North Pole 90° positive

albedo = 0.3

rad_surf = sol_rad_tilt_surf(donne, surface_orientation, albedo)

rad_surf.plot()
plt.xlabel("Time")
plt.ylabel("Solar irradiance,  Φ / (W·m⁻²)")
plt.show()

## Calcul de la radiation solaire arrivant sur le mur ##
β = surface_orientation['slope']
γ = surface_orientation['azimuth']
ϕ = surface_orientation['latitude']

# Transform degrees in radians
β = β * np.pi / 180
γ = γ * np.pi / 180
ϕ = ϕ * np.pi / 180

n = weather_data.index.dayofyear

declination_angle = 23.45 * np.sin(360 * (284 + n) / 365 * np.pi / 180)
δ = declination_angle * np.pi / 180

hour = weather_data.index.hour
minute = weather_data.index.minute + 60
hour_angle = 15 * ((hour + minute / 60) - 12)   # deg
ω = hour_angle * np.pi / 180                    # rad

theta = np.sin(δ) * np.sin(ϕ) * np.cos(β) \
    - np.sin(δ) * np.cos(ϕ) * np.sin(β) * np.cos(γ) \
    + np.cos(δ) * np.cos(ϕ) * np.cos(β) * np.cos(ω) \
    + np.cos(δ) * np.sin(ϕ) * np.sin(β) * np.cos(γ) * np.cos(ω) \
    + np.cos(δ) * np.sin(β) * np.sin(γ) * np.sin(ω)

theta = np.array(np.arccos(theta))
theta = np.minimum(theta, np.pi / 2)

dir_rad = weather_data["dir_n_rad"] * np.cos(theta)
dir_rad[dir_rad < 0] = 0

dif_rad = weather_data["dif_h_rad"] * (1 + np.cos(β)) / 2
