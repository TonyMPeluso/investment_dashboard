import json
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pulp
import webbrowser
import threading
from collections import OrderedDict
from pathlib import Path

# --- LP Solver ---
def solve_lp(df, objective_type, weights, discount_rate, budget_cap, ghg_target, reliability_target):
    df = df.copy()
    df["LCOE_$/kW"] *= (1 + discount_rate / 100)

    model = pulp.LpProblem("Investment_Optimization", pulp.LpMinimize)

    capacity_vars = {
        row["Technology"]: pulp.LpVariable(f"Cap_{row['Technology']}", 0, row["Capacity_Limit_MW"])
        for _, row in df.iterrows()
    }

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
    elif objective_type == "reliability":
        model += -pulp.lpSum([
            var * df.loc[df["Technology"] == tech, "Reliability_Score"].values[0]
            for tech, var in capacity_vars.items()
        ])
    elif objective_type == "weighted":
        model += pulp.lpSum([
            var * (
                weights["cost"] * df.loc[df["Technology"] == tech, "LCOE_$/kW"].values[0] * 1000 -
                weights["ghg"] * df.loc[df["Technology"] == tech, "GHG_Reduction_tCO2_per_kW"].values[0] * 1000 -
                weights["reliability"] * df.loc[df["Technology"] == tech, "Reliability_Score"].values[0]
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


# --- Scenario file setup ---
scenario_file = Path("data/scenarios.json")
scenario_file.parent.mkdir(parents=True, exist_ok=True)
if not scenario_file.exists():
    scenario_file.write_text("[]")


# --- UI ---
app_ui = ui.page_fluid(
    ui.input_radio_buttons("objective_type", "Objective Function",
        choices=["cost", "ghg", "reliability", "weighted"],
        selected="cost"),
    ui.input_numeric("weight_cost", "Weight: Cost", 0.5),
    ui.input_numeric("weight_ghg", "Weight: GHG", 0.3),
    ui.input_numeric("weight_reliability", "Weight: Reliability", 0.2),
    ui.input_numeric("discount_rate", "Discount Rate (%)", 5.0, min=0, max=20, step=0.1),
    ui.input_select("demand_scenario", "Demand Growth Scenario",
        choices=["Low", "Medium", "High"], selected="Medium"),
    ui.h2("Investment Optimization Dashboard"),
    ui.layout_columns(
        ui.card(ui.h4("Summary Results"), ui.output_table("summary_table")),
        ui.card(ui.h4("Contributions of Investments"), ui.output_table("detailed_table")),
        col_widths=[4, 8]
    ),
    ui.input_text("scenario_name", "Scenario Name", ""),
    ui.input_action_button("save_scenario", "Save Scenario"),
    ui.card(ui.h4("Saved Scenarios"), ui.output_table("saved_scenarios")),
    ui.layout_columns(
        ui.card(ui.output_plot("supply_stack")),
        ui.card(ui.output_plot("abatement_curve"))
    )
)


# --- Server ---
def server(input, output, session):

    @reactive.Calc
    def df():
        raw_df = pd.read_csv("data/tech_parameters.csv")
        return solve_lp(
            raw_df,
            input.objective_type(),
            {
                "cost": input.weight_cost(),
                "ghg": input.weight_ghg(),
                "reliability": input.weight_reliability()
            },
            input.discount_rate(),
            5_000_000_000,
            2_000_000,
            2000
        )

    @output
    @render.table
    def summary_table():
        d = df()
        summary = pd.DataFrame({
            "Metric": ["Total Investment (MW)", "Total Cost ($)", "Total GHG Saved (tCO₂e)", "Total Reliability"],
            "Value": [
                f"{int(d['Investment_MW'].sum()):,}",
                f"{int(d['Total_Cost_$'].sum()):,}",
                f"{int(d['Total_GHG_tCO2'].sum()):,}",
                f"{int(d['Reliability_Contribution'].sum()):,}"
            ]
        })
        return summary.style.set_properties(**{"text-align": "right"}).hide(axis="index")

    @output
    @render.table
    def detailed_table():
        d = df().copy()
        for col in ["Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"]:
            d[col] = d[col].apply(lambda x: f"{int(x):,}")
        return d[["Technology", "Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"]].style.set_properties(**{"text-align": "right"}).hide(axis="index")

    @output
    @render.plot
    def supply_stack():
        d = df()
        tech_list = sorted(d["Technology"].unique())
        palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        color_map = OrderedDict({tech: palette[i % len(palette)] for i, tech in enumerate(tech_list)})

        sorted_df = d.sort_values("LCOE_$/kW").copy()
        sorted_df["Block_Start"] = sorted_df["Investment_MW"].cumsum() - sorted_df["Investment_MW"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for _, row in sorted_df.iterrows():
            ax.bar(x=row["Block_Start"], height=row["LCOE_$/kW"], width=row["Investment_MW"],
                   color=color_map[row["Technology"]], align="edge", edgecolor="black", label=row["Technology"])
        ax.set_title("Power Supply Stack: LCOE vs. Cumulative Capacity")
        ax.set_xlabel("Cumulative Capacity (MW)")
        ax.set_ylabel("LCOE ($/kW)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def abatement_curve():
        d = df().copy()
        d["Abatement_Cost_per_tCO2"] = d["Total_Cost_$"] / d["Total_GHG_tCO2"]
        tech_list = sorted(d["Technology"].unique())
        palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        color_map = OrderedDict({tech: palette[i % len(palette)] for i, tech in enumerate(tech_list)})

        sorted_df = d.sort_values("Abatement_Cost_per_tCO2").copy()
        sorted_df["Block_Start"] = sorted_df["Total_GHG_tCO2"].cumsum() - sorted_df["Total_GHG_tCO2"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for _, row in sorted_df.iterrows():
            ax.bar(x=row["Block_Start"], height=row["Abatement_Cost_per_tCO2"], width=row["Total_GHG_tCO2"],
                   color=color_map[row["Technology"]], align="edge", edgecolor="black", label=row["Technology"])
        ax.set_title("GHG Abatement Curve: Cost per Tonne vs. Cumulative Abatement")
        ax.set_xlabel("Cumulative GHG Abated (tCO₂e)")
        ax.set_ylabel("Cost per tCO₂e ($/tonne)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig


# --- Launch App ---
app = App(app_ui, server)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
