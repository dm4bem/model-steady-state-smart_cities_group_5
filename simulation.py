import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from modelling import TC
from inputs import input_data_set
import dm4bem


controller = True
Kp = 1e3    # W/°C, controller gain

neglect_air_capacity = False
neglect_glass_capacity = False

explicit_Euler = True

imposed_time_step = True # umzustellen !!!!!!!!!!!!!
Δt = 3600    # s, imposed time step 

# MODEL
# =====
# Thermal circuits
dm4bem.print_TC(TC)

# by default TC['G']['q7'] = 0 # Kp -> 0, no controller (free-floating)
if controller:
    Kp = TC['G']['q7']     # G11 = Kp, conductance of edge q7
                            # Kp -> ∞, almost perfect controller
print('Kp:', Kp)

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)    # max time step for Euler explicit stability
dt = dm4bem.round_time(dtmax)

if imposed_time_step:
    dt = Δt

dm4bem.print_rounded_time('dt', dt)

# INPUT DATA SET
# ============== Imported from inputs.py

# interpolate the data from hourly to our time step (dt = 3s)
input_data_set = input_data_set.resample(
    str(dt) + 'S').interpolate(method='linear')

# Input vector in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ0 = 20.0                   # °C, initial temperatures
θ = pd.DataFrame(index=u.index)
θ[As.columns] = θ0          # fill θ with initial valeus θ0

# Time integration, ss model integrated in time using Euler forward (explicit) method
I = np.eye(As.shape[0])     # identity matrix

if explicit_Euler:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = (I + dt * As) @ θ.iloc[k] + dt * Bs @ u.iloc[k] # @ is the matrix multiplication operator
else:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = np.linalg.inv(
            I - dt * As) @ (θ.iloc[k] + dt * Bs @ u.iloc[k])
        
# Outputs
y = (Cs @ θ.T + Ds @  u.T).T

S = 5*4                   # m², surface area of the building
q_HVAC = Kp * (u['q7'] - y['θ5']) / S  # W/m²
y['θ5']

### PLOTS ###

data = pd.DataFrame({'To': input_data_set['To'],
                     'θi': y['θ5'],
                     'Etot': input_data_set['Etot'],
                     'q_HVAC': q_HVAC})

fig, axs = plt.subplots(2, 1)
data[['To', 'θi']].plot(ax=axs[0],
                        xticks=[],
                        ylabel='Temperature, $θ$ / °C')

axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
              loc='upper right')

data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                              ylabel='Heat rate, $q$ / (W·m⁻²)')
axs[1].set(xlabel='Time')
axs[1].legend(['$E_{total}$', '$q_{HVAC}$'],
              loc='upper right')
plt.show()
