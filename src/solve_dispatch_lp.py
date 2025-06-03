import pandas as pd
import numpy as np
import pulp


def solve_dispatch_lp(load_curve_file: str, dispatch_profiles_file: str):
    # Load demand
    load = pd.read_csv(load_curve_file)
    demand = load["Load_MW"].values
    hours = list(range(24))

    # Load dispatch profiles
    dispatch_df = pd.read_csv(dispatch_profiles_file)

    technologies = dispatch_df["Technology"].unique()
    model = pulp.LpProblem("Daily_Dispatch", pulp.LpMinimize)

    # Create dispatch variables for each technology and hour
    dispatch_vars = {
        (tech, h): pulp.LpVariable(f"{tech}_{h}", lowBound=0)
        for tech in technologies for h in hours
    }

    # Create battery and pumped hydro variables (charge and discharge)
    for storage in ["Battery", "Pumped_Hydro"]:
        for h in hours:
            dispatch_vars[(f"{storage}_charge", h)] = pulp.LpVariable(f"{storage}_charge_{h}", lowBound=0)
            dispatch_vars[(f"{storage}_discharge", h)] = pulp.LpVariable(f"{storage}_discharge_{h}", lowBound=0)

    # Objective: Minimize total cost
    model += pulp.lpSum(
        dispatch_vars[(tech, h)] * dispatch_df.loc[dispatch_df["Technology"] == tech, "LCOE_$/kWh"].values[0]
        for tech in technologies for h in hours
    )

    # Demand constraint: supply must meet hourly demand
    for h in hours:
        supply = pulp.lpSum(dispatch_vars[(tech, h)] for tech in technologies)
        supply += dispatch_vars[("Battery_discharge", h)] + dispatch_vars[("Pumped_Hydro_discharge", h)]
        supply -= dispatch_vars[("Battery_charge", h)] + dispatch_vars[("Pumped_Hydro_charge", h)]
        model += supply == demand[h], f"Demand_Constraint_{h}"

    # Energy balance: net charge/discharge must be zero over 24 hours
    for storage in ["Battery", "Pumped_Hydro"]:
        charge = pulp.lpSum(dispatch_vars[(f"{storage}_charge", h)] for h in hours)
        discharge = pulp.lpSum(dispatch_vars[(f"{storage}_discharge", h)] for h in hours)
        model += charge == discharge, f"{storage}_NetZero"

    # Solve the model
    model.solve()

    # Compile results
    results = []
    for h in hours:
        row = {"Hour": h, "Demand_MW": demand[h]}
        for tech in technologies:
            row[tech] = dispatch_vars[(tech, h)].varValue
        row["Battery_charge"] = dispatch_vars[("Battery_charge", h)].varValue
        row["Battery_discharge"] = dispatch_vars[("Battery_discharge", h)].varValue
        row["Pumped_Hydro_charge"] = dispatch_vars[("Pumped_Hydro_charge", h)].varValue
        row["Pumped_Hydro_discharge"] = dispatch_vars[("Pumped_Hydro_discharge", h)].varValue
        results.append(row)

    return pd.DataFrame(results)
