import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

## Building properties
l = 4               # m length of the cubic room
hight = 3               # m height of the walls
Sw = hight * 1.30       # m² surface area of one window
Sd = hight * 1          # m² surface area of the door
Sc = Si =  4 * l * hight - Sw - Sd   # m² surface area of concrete & insulation of the walls

## Thermophysical properties

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': 5 * l**2}            # m²

insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': 5 * l**2}          # m²

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': l**2}                   # m²

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')


# radiative properties
ε_wLW = 0.85    # long wave emmissivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmissivity: glass pyrex
ε_dLW = 0.82    # long wave emmissivity: wood (general hardwood, Engineering Toolbox)
α_dSW = 0.8     # short wave absortivity: wood brown (Wufiwiki)
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmittance: reflective blue glass


σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
print(f'σ = {σ} W/(m²⋅K⁴)')
# Convection coefficient
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)

