import pulp
import pandas as pd

def solve_dispatch_lp(supply_df, demand_series, battery_df, pumped_df, smart_grid_series, net_metering_series):
    """
    Solve the dispatch LP given supply and demand profiles.

    Parameters:
        supply_df (pd.DataFrame): DataFrame with columns:
            - 'Technology': name
            - 'Profile': list of 24 floats (availability per hour)
            - 'Cost': scalar cost per MWh
        demand_series (pd.Series): hourly demand (24 values)
        battery_df (dict): contains:
            - 'max_charge_rate', 'max_discharge_rate', 'efficiency', 'cost'
        pumped_df (dict): same as battery_df
        smart_grid_series (pd.Series): hourly demand reduction
        net_metering_series (pd.Series): hourly behind-the-meter solar generation

    Returns:
        dict: containing LP results, dispatch schedule, and total cost.
    """

    hours = range(24)
    techs = supply_df['Technology'].tolist()
    model = pulp.LpProblem("Dispatch_Optimization", pulp.LpMinimize)

    # Variables: Dispatch for each tech and hour
    dispatch = pulp.LpVariable.dicts("Dispatch", ((t, h) for t in techs for h in hours), lowBound=0)
    battery_charge = pulp.LpVariable.dicts("Battery_Charge", hours, lowBound=0)
    battery_discharge = pulp.LpVariable.dicts("Battery_Discharge", hours, lowBound=0)
    pumped_charge = pulp.LpVariable.dicts("Pumped_Charge", hours, lowBound=0)
    pumped_discharge = pulp.LpVariable.dicts("Pumped_Discharge", hours, lowBound=0)

    # Objective: Minimize total cost
    model += pulp.lpSum([
        dispatch[t, h] * supply_df.loc[supply_df['Technology'] == t, 'Cost'].values[0]
        for t in techs for h in hours
    ]) + pulp.lpSum([
        battery_discharge[h] * battery_df['cost'] + pumped_discharge[h] * pumped_df['cost']
        for h in hours
    ])

    for h in hours:
        # Total supply = demand - DSM (smart grid + net metering)
        net_demand = demand_series[h] - smart_grid_series[h] - net_metering_series[h]

        supply = pulp.lpSum([dispatch[t, h] for t in techs])
        storage_discharge = battery_discharge[h] + pumped_discharge[h]
        storage_charge = battery_charge[h] + pumped_charge[h]

        model += supply + storage_discharge == net_demand + storage_charge, f"Balance_hour_{h}"

        # Supply tech availability
        for t in techs:
            availability = supply_df.loc[supply_df['Technology'] == t, 'Profile'].values[0][h]
            model += dispatch[t, h] <= availability, f"Availability_{t}_{h}"

        # Battery and pumped constraints
        model += battery_charge[h] <= battery_df['max_charge_rate'], f"Battery_charge_{h}"
        model += battery_discharge[h] <= battery_df['max_discharge_rate'], f"Battery_discharge_{h}"

        model += pumped_charge[h] <= pumped_df['max_charge_rate'], f"Pumped_charge_{h}"
        model += pumped_discharge[h] <= pumped_df['max_discharge_rate'], f"Pumped_discharge_{h}"

    # Energy balance for battery and pumped hydro over the day
    model += pulp.lpSum([battery_charge[h] * battery_df['efficiency'] for h in hours]) == \
             pulp.lpSum([battery_discharge[h] for h in hours]), "Battery_energy_balance"

    model += pulp.lpSum([pumped_charge[h] * pumped_df['efficiency'] for h in hours]) == \
             pulp.lpSum([pumped_discharge[h] for h in hours]), "Pumped_energy_balance"

    model.solve()

    # Return results
    results = {
        "dispatch": {(t, h): dispatch[t, h].varValue for t in techs for h in hours},
        "battery_charge": [battery_charge[h].varValue for h in hours],
        "battery_discharge": [battery_discharge[h].varValue for h in hours],
        "pumped_charge": [pumped_charge[h].varValue for h in hours],
        "pumped_discharge": [pumped_discharge[h].varValue for h in hours],
        "total_cost": pulp.value(model.objective)
    }

    return results