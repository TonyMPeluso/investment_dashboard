import pandas as pd
import pulp

# Load the parameter table
param_path = "data/tech_parameters.csv"
tech_params = pd.read_csv(param_path)

# Define LP model
model = pulp.LpProblem("Investment_Optimization", pulp.LpMinimize)

# Objective selector
objective_type = "cost"  # options: 'cost', 'ghg', 'reliability', 'weighted'
weights = {"cost": 0.5, "ghg": 0.3, "reliability": 0.2}  # used if 'weighted'

# Constraints
budget_cap = 5_000_000_000      # dollars
ghg_target = 2_000_000          # tonnes CO2e
reliability_target = 2_000      # arbitrary units

# Decision variables: how much to invest in each tech (MW)
capacity_vars = {
    row["Technology"]: pulp.LpVariable(f"Cap_{row['Technology']}", lowBound=0, upBound=row["Capacity_Limit_MW"])
    for _, row in tech_params.iterrows()
}

# Objective function
if objective_type == "cost":
    model += pulp.lpSum([
        var * tech_params.loc[tech_params["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000
        for tech, var in capacity_vars.items()
    ])
elif objective_type == "ghg":
    model += -pulp.lpSum([
        var * tech_params.loc[tech_params["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000
        for tech, var in capacity_vars.items()
    ])
elif objective_type == "reliability":
    model += -pulp.lpSum([
        var * tech_params.loc[tech_params["Technology"] == tech, "Reliability_Score"].values[0]
        for tech, var in capacity_vars.items()
    ])
elif objective_type == "weighted":
    model += pulp.lpSum([
        var * (
            weights["cost"] * tech_params.loc[tech_params["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000 -
            weights["ghg"] * tech_params.loc[tech_params["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000 -
            weights["reliability"] * tech_params.loc[tech_params["Technology"] == tech, "Reliability_Score"].values[0]
        )
        for tech, var in capacity_vars.items()
    ])

# Constraint: total cost ≤ budget
model += pulp.lpSum([
    var * tech_params.loc[tech_params["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000
    for tech, var in capacity_vars.items()
]) <= budget_cap

# Constraint: GHG savings ≥ target
model += pulp.lpSum([
    var * tech_params.loc[tech_params["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000
    for tech, var in capacity_vars.items()
]) >= ghg_target

# Constraint: reliability contribution ≥ target
model += pulp.lpSum([
    var * tech_params.loc[tech_params["Technology"] == tech, "Reliability_Score"].values[0]
    for tech, var in capacity_vars.items()
]) >= reliability_target

# Solve LP
model.solve()

# Add investment results back to table
tech_params["Investment_MW"] = tech_params["Technology"].apply(lambda tech: capacity_vars[tech].varValue)
tech_params["Total_Cost_$"] = tech_params["Investment_MW"] * tech_params["LCOE_$/kW"] * 1000
tech_params["Total_GHG_tCO2"] = tech_params["Investment_MW"] * tech_params["GHG_Reduction_tCO2_per_kW"] * 1000
tech_params["Reliability_Contribution"] = tech_params["Investment_MW"] * tech_params["Reliability_Score"]

# Print results
print("\nDetailed Results:")
print(tech_params[["Technology", "Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"]])

print("\nSummary:")
print("Total Investment (MW):", tech_params["Investment_MW"].sum())
print("Total Cost ($):", tech_params["Total_Cost_$"].sum())
print("Total GHG Saved (tCO₂e):", tech_params["Total_GHG_tCO2"].sum())
print("Total Reliability:", tech_params["Reliability_Contribution"].sum())
