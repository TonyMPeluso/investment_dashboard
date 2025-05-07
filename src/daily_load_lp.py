import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

# --- STEP 1: Define Inputs ---
hours = np.arange(24)

# Winter peak demand profile
demand_step = np.array([
    2400, 2250, 2220, 2200, 2220, 2300, 2600, 2750,
    2850, 2850, 2900, 2950, 3000, 3050, 3100, 3100,
    3400, 3600, 3850, 3850, 3450, 3050, 2850, 2500
])

# Solar (sine curve from 7 to 17)
solar_profile = np.zeros(24)
solar_hours = np.arange(7, 18)
solar_profile[solar_hours] = np.sin(np.pi * (solar_hours - 7) / (17 - 7))
solar_output = solar_profile * 1000

# Wind (offshore pattern)
wind_profile = 0.4 + 0.4 * np.sin((np.pi / 16) * (hours - 5))**2
wind_output = wind_profile * 1000

# --- STEP 2: Set Up LP Model ---
model = pulp.LpProblem("DispatchOptimization", pulp.LpMinimize)

# Cost assumptions ($/MWh)
costs = {
    "Solar": 60,
    "Wind": 55,
    "Hydro": 30,
    "Battery_Discharge": 150,
    "Pumped_Discharge": 120,
    "Net_Metering": 65,
    "Smart_Grid": 90
}

# Define LP variables
solar = pulp.LpVariable.dicts("Solar", hours, 0)
wind = pulp.LpVariable.dicts("Wind", hours, 0)
hydro = pulp.LpVariable.dicts("Hydro", hours, 0)
battery_dch = pulp.LpVariable.dicts("Battery_Discharge", hours, 0)
battery_ch = pulp.LpVariable.dicts("Battery_Charge", hours, 0)
pumped_dch = pulp.LpVariable.dicts("Pumped_Discharge", hours, 0)
pumped_ch = pulp.LpVariable.dicts("Pumped_Charge", hours, 0)
netmeter = pulp.LpVariable.dicts("Net_Metering", hours, 0)
smart = pulp.LpVariable.dicts("Smart_Grid", hours, 0)
battery_E = [pulp.LpVariable(f"Battery_E_{t}", 0, 800) for t in hours]
pumped_E = [pulp.LpVariable(f"Pumped_E_{t}", 0, 1600) for t in hours]

# Objective: minimize cost
model += pulp.lpSum([
    costs["Solar"] * solar[t] +
    costs["Wind"] * wind[t] +
    costs["Hydro"] * hydro[t] +
    costs["Battery_Discharge"] * battery_dch[t] +
    costs["Pumped_Discharge"] * pumped_dch[t] +
    costs["Net_Metering"] * netmeter[t] +
    costs["Smart_Grid"] * smart[t]
    for t in hours
])

# Constraints
for t in hours:
    model += solar[t] <= solar_output[t]
    model += wind[t] <= wind_output[t]
    model += hydro[t] <= 2000
    model += battery_dch[t] <= 100
    model += battery_ch[t] <= 100
    model += pumped_dch[t] <= 200
    model += pumped_ch[t] <= 200
    model += smart[t] <= 0.05 * demand_step[t]
    model += netmeter[t] <= 0.3 * solar_output[t]

    # Demand constraint
    model += (
        solar[t] + wind[t] + hydro[t] +
        battery_dch[t] + pumped_dch[t] +
        netmeter[t] + smart[t]
    ) >= demand_step[t]

# Storage balances
for t in hours:
    if t == 0:
        model += battery_E[t] == battery_E[-1] + battery_ch[t] - battery_dch[t]
        model += pumped_E[t] == pumped_E[-1] + pumped_ch[t] - pumped_dch[t]
    else:
        model += battery_E[t] == battery_E[t - 1] + battery_ch[t] - battery_dch[t]
        model += pumped_E[t] == pumped_E[t - 1] + pumped_ch[t] - pumped_dch[t]

# --- STEP 3: Solve LP ---
model.solve()

# --- STEP 4: Extract Results ---
df = pd.DataFrame({
    "Hour": hours,
    "Demand": demand_step,
    "Solar": [solar[t].varValue for t in hours],
    "Wind": [wind[t].varValue for t in hours],
    "Hydro": [hydro[t].varValue for t in hours],
    "Battery_Discharge": [battery_dch[t].varValue for t in hours],
    "Pumped_Discharge": [pumped_dch[t].varValue for t in hours],
    "Net_Metering": [netmeter[t].varValue for t in hours],
    "Smart_Grid": [smart[t].varValue for t in hours]
})

df["Net Supply"] = df[
    ["Solar", "Wind", "Hydro", "Battery_Discharge",
     "Pumped_Discharge", "Net_Metering", "Smart_Grid"]
].sum(axis=1)

# Available capacity
available_capacity = np.minimum(solar_output, 1000) + \
                     np.minimum(wind_output, 1000) + \
                     2000 + 100 + 200 + \
                     0.3 * solar_output + 0.05 * demand_step

# --- STEP 5: Plot ---
colors = ['gold', 'skyblue', 'navy', 'orchid', 'seagreen', 'darkorange', 'darkred']
sources = ["Solar", "Wind", "Hydro", "Battery_Discharge",
           "Pumped_Discharge", "Net_Metering", "Smart_Grid"]

plt.figure(figsize=(14, 7))
bottom = np.zeros(24)
for src, col in zip(sources, colors):
    plt.bar(df["Hour"], df[src], bottom=bottom, label=src, color=col, width=0.7)
    bottom += df[src]

plt.step(df["Hour"], df["Demand"], label="Demand", color="black", linewidth=2, where="mid")
plt.plot(df["Hour"], available_capacity, label="Available Capacity", color="red", linestyle='--', linewidth=2)

plt.xlabel("Hour of Day")
plt.ylabel("Power (MW)")
plt.title("Dispatch with Demand and Available Capacity (LP-Enforced)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
