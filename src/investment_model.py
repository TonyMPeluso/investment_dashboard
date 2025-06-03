from shiny import App, ui, render, reactive, req
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict
from solve_investment_lp import solve_investment_lp
from solve_dispatch_lp import solve_dispatch_lp

# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Investment Optimization Dashboard"),
    ui.navset_tab(
        ui.nav("Investment Results",
            ui.layout_columns(
                ui.input_radio_buttons("objective_type", "Objective Function",
                    choices=["cost", "ghg", "weighted"], selected="cost"),
                ui.input_slider("weight_cost", "Cost Weight", min=0, max=1, value=0.5, step=0.01),
                col_widths=[3, 9]
            ),
            ui.input_numeric("discount_rate", "Discount Rate (%)", 5.0, min=0, max=20, step=0.1),
            ui.input_select("demand_scenario", "Power Demand Growth Scenario",
                choices=["Low", "Medium", "High"], selected="Medium"),
            ui.layout_columns(
                ui.input_text("scenario_name", "Scenario Name", ""),
                ui.input_action_button("run_button", "Run Scenario"),
                col_widths=[8, 4]
            ),
            ui.layout_columns(
                ui.card(ui.h4("Summary Results"), ui.output_table("summary_table")),
                ui.card(ui.h4("Contributions of Investments"), ui.output_table("detailed_table")),
                col_widths=[4, 8]
            ),
            ui.layout_columns(
                ui.card(ui.output_plot("supply_stack")),
                ui.card(ui.output_plot("abatement_curve"))
            )
        ),
        ui.nav("Dispatch Results",
            ui.card(ui.output_plot("dispatch_plot"))
        )
    )
)

# --- Server ---
def server(input, output, session):
    @reactive.Calc
    def investment_df():
        req(input.run_button())
        tech_df = pd.read_csv("data/tech_parameters.csv")
        return solve_investment_lp(
            tech_df,
            input.objective_type(),
            input.weight_cost(),
            input.discount_rate(),
            5_000_000_000,
            2_000_000,
            2000
        )

    @reactive.Calc
    def dispatch_df():
        demand_file = f"data/load_curve_winter_{input.demand_scenario().lower()}.csv"
        return solve_dispatch_lp(investment_df(), demand_file)

    @reactive.effect
    def _():
        session.send_input_message("weight_cost", {"disabled": input.objective_type() != "weighted"})

    @output
    @render.table
    def summary_table():
        d = investment_df()
        summary = pd.DataFrame({
            "Metric": ["Total Investment (MW)", "Total Cost ($)", "Total GHG Saved (tCO₂e)", "Total Reliability"],
            "Value": [
                f"{int(d['Investment_MW'].sum()):,}",
                f"{int(d['Total_Cost_$'].sum()):,}",
                f"{int(d['Total_GHG_tCO2'].sum()):,}",
                f"{int(d['Reliability_Contribution'].sum()):,}"
            ]
        })
        return summary.style.set_properties(subset=["Metric"], **{"text-align": "left"}) \
                            .set_properties(subset=["Value"], **{"text-align": "right"}) \
                            .hide(axis="index")

    @output
    @render.table
    def detailed_table():
        d = investment_df().copy()
        for col in ["Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"]:
            d[col] = d[col].apply(lambda x: f"{int(x):,}")
        return d[["Technology", "Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"]] \
            .style.set_properties(subset=["Technology"], **{"text-align": "left"}) \
                .set_properties(subset=["Investment_MW", "Total_Cost_$", "Total_GHG_tCO2", "Reliability_Contribution"],
                                **{"text-align": "right"}) \
                .hide(axis="index")

    @output
    @render.plot
    def supply_stack():
        d = investment_df()
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
        d = investment_df().copy()
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

    @output
    @render.plot
    def dispatch_plot():
        d = dispatch_df()
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = pd.Series([0]*len(d))
        supply_cols = d.columns.drop("Hour")
        for tech in supply_cols:
            ax.bar(d["Hour"], d[tech], bottom=bottom, label=tech)
            bottom += d[tech]
        ax.plot(d["Hour"], d[supply_cols].sum(axis=1), label="Total Supply", color="black", linewidth=1.5)
        ax.set_title("Hourly Dispatch: Supply by Technology")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("MW Supplied")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        return fig

app = App(app_ui, server)
