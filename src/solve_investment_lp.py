import pandas as pd
import pulp

def solve_investment_lp(df, objective_type, weight_cost, discount_rate, budget_cap, ghg_target, reliability_target):
    df = df.copy()
    df["LCOE_$/kW"] *= (1 + discount_rate / 100)

    model = pulp.LpProblem("Investment_Optimization", pulp.LpMinimize)

    capacity_vars = {
        row["Technology"]: pulp.LpVariable(f"Cap_{row['Technology']}", 0, row["Capacity_Limit_MW"])
        for _, row in df.iterrows()
    }

    weight_ghg = 1 - weight_cost

    if objective_type == "cost":
        model += pulp.lpSum([
            var * df.loc[df["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000
            for tech, var in capacity_vars.items()
        ])
    elif objective_type == "ghg":
        model += -pulp.lpSum([
            var * df.loc[df["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000
            for tech, var in capacity_vars.items()
        ])
    elif objective_type == "weighted":
        model += pulp.lpSum([
            var * (
                weight_cost * df.loc[df["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000 -
                weight_ghg * df.loc[df["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000
            )
            for tech, var in capacity_vars.items()
        ])

    model += pulp.lpSum([
        var * df.loc[df["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000
        for tech, var in capacity_vars.items()
    ]) <= budget_cap

    model += pulp.lpSum([
        var * df.loc[df["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000
        for tech, var in capacity_vars.items()
    ]) >= ghg_target

    model += pulp.lpSum([
        var * df.loc[df["Technology"] == tech, "Reliability_Score"].values[0]
        for tech, var in capacity_vars.items()
    ]) >= reliability_target

    model.solve()

    df["Investment_MW"] = df["Technology"].apply(lambda tech: capacity_vars[tech].varValue)
    df["Total_Cost_$"] = df["Investment_MW"] * df["LCOE_$/kW"] * 1000
    df["Total_GHG_tCO2"] = df["Investment_MW"] * df["GHG_Reduction_tCO2_per_kW"] * 1000
    df["Reliability_Contribution"] = df["Investment_MW"] * df["Reliability_Score"]

    return df