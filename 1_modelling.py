import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

## Building properties
l = 4               # m length of the cubic room
Sg = l**2           # m² surface area of one window
Sc = Si = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])
dddddd
