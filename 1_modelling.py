import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

## Building properties
l = 4               # m length of the cubic room
Sg = l**2           # m² surface area of one window
Sc = Si = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls

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

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

