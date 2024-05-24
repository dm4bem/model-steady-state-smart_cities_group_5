import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

## Building properties
a = 4               # m length 
b = 5
hight = 3               # m height of the walls
Sw = hight * 1.30       # m² surface area of one window
Sd = hight * 1          # m² surface area of the door
Sc = Si =  4 * a * b * hight - 2*Sw - Sd   # m² surface area of concrete & insulation of the walls

## Thermophysical properties

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': Sc}                  # m²

insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': Sc}                # m²

glass = {'Conductivity': 2.8,               # W/(m²·K) - https://www.gov.scot/binaries/content/documents/govscot/publications/advice-and-guidance/2020/02/tables-of-u-values-and-thermal-conductivity/documents/6-a---tables-of-u-values-and-thermal-conductivity/6-a---tables-of-u-values-and-thermal-conductivity/govscot%3Adocument/6.A%2B-%2BTables%2Bof%2BU-values%2Band%2Bthermal%2Bconductivity%2B%2B.pdf
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 1,                        # m - 1m because m² at conductivity
         'Surface': Sw}                     # m²

wood = {'Conductivity': 0.14,               # W/(m·K) - https://materials.ads.org.uk/larch/
         'Density': 500,                    # kg/m³ - Larch wood; EngineeringToolbox.com
         'Specific heat': 1500,             # J/(kg⋅K) - Larch wood; https://web.ornl.gov/sci/buildings/conf-archive/1992%20B5%20papers/028.pdf#:~:text=URL%3A%20https%3A%2F%2Fweb.ornl.gov%2Fsci%2Fbuildings%2Fconf
         'Width': 0.05,                     # m
         'Surface': Sd}                     # m²

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass, 
                               'Wood': wood},
                              orient='index')


# radiative properties
ε_wLW = 0.85    # long wave emmissivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmissivity: glass pyrex
ε_dLW = 0.82    # long wave emmissivity: wood (general hardwood, EngineeringToolbox.com)
α_dSW = 0.8     # short wave absortivity: wood brown (Wufiwiki)
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmittance: reflective blue glass


σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
print(f'σ = {σ} W/(m²⋅K⁴)')
# Convection coefficient
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)

###
### Thermal Network
# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
print(G_cd)
pd.DataFrame(G_cd, columns=['Conductance'])

# convection
Gw = h * wall['Surface'].iloc[0]     # wall
Gg = h * wall['Surface'].iloc[2]     # glass
Gd = h * wall['Surface'].iloc[3]     # wood

# view factor wall-glass
    # our windows are in the same plane - no radiation
Fwg = glass['Surface'] / concrete['Surface']
# view factor wall-wood
Fwd = wood['Surface'] / concrete['Surface']

# long wave radiation negligible

####
## Advection 

# ventilation flow rate
Va = a*b*hight              # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration
Kp = 0 # no controller? 

## Thermal capacities
C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']

C1 = C[0]
print('concrete cap: ', C1)
C3 = C[1]
print('insul cap: ', C3)
C5 = air['Density'] * air['Specific heat'] * Va
print('air cap: ', C5)




## MATRICES 

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

A = np.zeros([8, 6])        # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = 0, 1     # branch 5: node 4 -> node 5
A[6, 4], A[6, 5] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5] = 1                 # branch 7: node 5 -> node 6

pd.DataFrame(A, index=q, columns=θ)
print('A: ', A)


C = np.array([0, C1, 0, C3, 0, C5])
pd.DataFrame(C, index=θ)
print('C: ', C)
