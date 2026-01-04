"""
Investment Optimization Dashboard — Main Application

This module defines the Shiny for Python application that:
- Loads technology, demand, and availability data
- Solves an investment + dispatch optimization problem
- Displays investment results, supply stacks, dispatch profiles,
  and abatement curves interactively

Author: Anthony Peluso
Project: Investment Optimization Dashboard
Last updated: 2025-01-XX
"""

# ============================
# Standard library imports
# ============================
import os
import sys
from collections import OrderedDict

# ============================
# Third-party imports
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from shiny import App, ui, render, reactive, req

# ============================
# Local application imports
# ============================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from solve_investment_dispatch_lp import solve_investment_dispatch_lp

# ============================================================
# Debug / display configuration
# ============================================================

# Turn on by running:
#   DEBUG=1 shiny run --reload src/investment_model.py
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "TRUE", "yes", "YES")

def fmt2(x) -> str:
    """Format numeric as xxx,xxx.yy (safe for NaN/None)."""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

# ============================================================
# Helpers
# ============================================================
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).copy()


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom and denom != 0 else 0.0


def _tech_color_map(tech_list):
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    return OrderedDict({tech: palette[i % len(palette)] for i, tech in enumerate(tech_list)})


def _debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# ============================================================
# UI
# ============================================================
SUPPLY_STACK_CAPTION = (
    "Supply stack ordered by Proxy LCOE. "
    "Block width = capacity (MW). Block height = Proxy LCOE ($/MWh). "
    "Imported Hydro is displayed as peak dispatched MW (not an investment)."
)

ABATEMENT_CAPTION = (
    "Marginal abatement cost (MAC) curve. "
    "Block width = daily abatement (tCO₂/day). Block height = MAC ($/tCO₂). "
    "MAC is computed as Proxy LCOE divided by the abatement factor (tCO₂/MWh)."
)

app_ui = ui.page_fluid(
    ui.h2("Investment Optimization Dashboard"),
    ui.navset_tab(
        ui.nav_panel(
            "Investment Results",

            # ---- Top controls: 3 columns ----
            ui.layout_columns(
                ui.card(
                    ui.h5("Objective"),
                    ui.input_radio_buttons(
                        "objective_type",
                        "Objective Function (UI only for now)",
                        choices=["cost", "ghg", "weighted"],
                        selected="cost",
                    ),
                    ui.input_slider(
                        "weight_cost",
                        "Cost Weight (UI only)",
                        min=0, max=1, value=0.5, step=0.01
                    ),
                ),
                ui.card(
                    ui.h5("Scenario Parameters"),
                    ui.input_numeric("discount_rate", "Discount Rate (%)", 5.0, min=0, max=20, step=0.1),
                    ui.input_numeric(
                        "max_ev_load_mw",
                        "Maximum hourly EV charging load (MW)",
                        300.0,
                        min=0,
                        max=5000,
                        step=10,
                    ),
                    ui.input_select(
                        "demand_scenario",
                        "Power Demand Growth Scenario",
                        choices=["Low", "Medium", "High"],
                        selected="Medium",
                    ),
                ),
                ui.card(
                    ui.h5("Run"),
                    ui.input_text("scenario_name", "Scenario Name", ""),
                    ui.input_action_button("run_button", "Run Scenario"),
                ),
                col_widths=[4, 4, 4],
            ),

            ui.layout_columns(
                ui.card(ui.h4("Summary Results"), ui.output_table("summary_table")),
                ui.card(ui.h4("Investments + LCOE Proxy"), ui.output_table("investment_table")),
                col_widths=[4, 8],
            ),

            ui.layout_columns(
                ui.card(
                    ui.output_plot("supply_stack"),
                    ui.tags.div(
                        ui.tags.small(SUPPLY_STACK_CAPTION),
                        style="margin-top: 6px; color: #666;"
                    ),
                ),
                ui.card(
                    ui.output_plot("abatement_curve"),
                    ui.tags.div(
                        ui.tags.small(ABATEMENT_CAPTION),
                        style="margin-top: 6px; color: #666;"
                    ),
                ),
                col_widths=[6, 6],
            ),
        ),

        ui.nav_panel(
            "Dispatch Results",
            ui.card(
                ui.output_plot("dispatch_plot"),
                style="min-height: 650px;"
            ),
            ui.layout_columns(
                ui.card(ui.h5("Dispatch totals (diagnostics)"), ui.output_table("dispatch_totals")),
                col_widths=[6],   # half width
            ),
        ),
    ),
)


# ============================================================
# Server
# ============================================================
def server(input, output, session):

    if DEBUG:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    # Disable weight slider unless "weighted" selected (UI-only right now)
    @reactive.effect
    def _():
        session.send_input_message("weight_cost", {"disabled": input.objective_type() != "weighted"})

    @reactive.Calc
    def scenario_inputs():
        req(input.run_button())
        demand_file = f"data/load_curve_winter_{input.demand_scenario().lower()}.csv"
        return {
            "tech_path": "data/tech_parameters.csv",
            "ev_path": "data/ev_load_profile.csv",
            "demand_path": demand_file,
            "max_ev_load_mw": float(input.max_ev_load_mw()),
            "discount_rate": float(input.discount_rate()) / 100.0,
            "scenario_name": (input.scenario_name() or "").strip(),
        }

    @reactive.Calc
    def model_results():
        s = scenario_inputs()

        tech_df = _read_csv(s["tech_path"])
        load_df = _read_csv(s["demand_path"])
        ev_profile = _read_csv(s["ev_path"])
        availability_df = _read_csv("data/resources_availability.csv")

        # Basic sanity checks (fail fast with clear message)
        req("Technology" in tech_df.columns)
        req("Hour" in load_df.columns and "Load_MW" in load_df.columns)
        req("Hour" in ev_profile.columns and "EV_Profile" in ev_profile.columns)
        req(set(["Hour", "Technology", "Availability"]).issubset(set(availability_df.columns)))

        inv_df, disp_df = solve_investment_dispatch_lp(
            tech_df=tech_df,
            load_df=load_df,
            ev_profile=ev_profile,
            availability_df=availability_df,
            max_ev_load_mw=float(input.max_ev_load_mw()),
            discount_rate=float(input.discount_rate()) / 100.0,
        )

        return {
            "tech_df": tech_df,
            "load_df": load_df,
            "ev_profile": ev_profile,
            "investment_df": inv_df,
            "dispatch_df": disp_df,
            "meta": s
        }

    def _investment_with_cost_proxy():
        r = model_results()
        tech_df = r["tech_df"].copy()
        inv = r["investment_df"].copy()
        disp = r["dispatch_df"].copy()

        # ---- Merge cost/emissions fields into investment table ----
        tech_cost = tech_df[[
            "Technology",
            "Var_Cost_$/MWh",
            "Capex_$/kW",
            "Dispatchability",
            "GHG_Reduction_tCO2_per_MWh",
        ]].copy()

        out = inv.merge(tech_cost, on="Technology", how="left")

        # ---- numeric coercion ----
        for c in ["Investment_MW", "Daily_Capex_$", "Upfront_Capex_$", "Var_Cost_$/MWh", "Capex_$/kW", "GHG_Reduction_tCO2_per_MWh"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        # ---- Imported Hydro: NOT an investment in the table ----
        is_hydro = out["Technology"].astype(str).str.strip().eq("Imported Hydro")
        out.loc[is_hydro, "Investment_MW"] = 0.0
        # also force capex fields to 0 for hydro "investment"
        for c in ["Upfront_Capex_$", "Daily_Capex_$"]:
            if c in out.columns:
                out.loc[is_hydro, c] = 0.0

        # ---- Daily capex factor ----
        discount_rate = float(r["meta"]["discount_rate"])  # decimal already
        lifetime_years = 20
        if discount_rate <= 0:
            crf = 1.0 / lifetime_years
        else:
            crf = (discount_rate * (1 + discount_rate) ** lifetime_years) / ((1 + discount_rate) ** lifetime_years - 1)
        capex_daily_factor = crf / 365.0

        # recompute Daily_Capex_$ if missing or all zero
        if "Daily_Capex_$" not in out.columns or float(out["Daily_Capex_$"].sum()) == 0.0:
            out["Daily_Capex_$"] = out["Investment_MW"] * out["Capex_$/kW"] * 1000.0 * capex_daily_factor

        # ============================================================
        # Utilization (MWh/day) from dispatch
        # ============================================================
        disp_cols = set(disp.columns)
        util_mwh = {}

        for tech in tech_df["Technology"].tolist():
            # 1) Tech appears directly as a dispatch column (e.g., Wind, Solar PV, Net Metering, Imported Hydro)
            if tech in disp_cols:
                util_mwh[tech] = float(pd.to_numeric(disp[tech], errors="coerce").fillna(0.0).sum())
                continue

            dispatchability = str(tech_df.loc[tech_df["Technology"] == tech, "Dispatchability"].iloc[0])

            # 2) Balancing resources: utilization = total discharge
            if dispatchability == "Balancing":
                col_dis = f"{tech} Discharge"
                util_mwh[tech] = float(pd.to_numeric(disp[col_dis], errors="coerce").fillna(0.0).sum()) if col_dis in disp_cols else 0.0
                continue

            # 3) Demand-side shift (if present): utilization = "Down"
            dmode = "0"
            if "DSM_Mode" in tech_df.columns:
                dmode = str(tech_df.loc[tech_df["Technology"] == tech, "DSM_Mode"].iloc[0])
            if dispatchability == "Demand-side" and dmode == "shift":
                col_down = f"{tech} Down"
                util_mwh[tech] = float(pd.to_numeric(disp[col_down], errors="coerce").fillna(0.0).sum()) if col_down in disp_cols else 0.0
                continue

            # 4) Otherwise: assume always-on deployment for utilization proxy
            mw = float(out.loc[out["Technology"] == tech, "Investment_MW"].fillna(0.0).iloc[0])
            util_mwh[tech] = mw * 24.0

        out["Utilized_MWh_per_day"] = out["Technology"].map(util_mwh).fillna(0.0)

        # Abatement (tCO2/day)
        out["Abatement_tCO2_per_day"] = out["Utilized_MWh_per_day"] * out["GHG_Reduction_tCO2_per_MWh"]

        # ============================================================
        # Proxy LCOE ($/MWh)
        # ============================================================
        def _proxy(row):
            util = float(row.get("Utilized_MWh_per_day", 0.0))
            daily_capex = float(row.get("Daily_Capex_$", 0.0))
            var_cost = float(row.get("Var_Cost_$/MWh", 0.0))

            # Hydro: capex=0, so proxy LCOE = variable cost (e.g., $70/MWh)
            if daily_capex <= 1e-9:
                return var_cost

            if util <= 1e-9:
                return 0.0

            return (daily_capex / util) + var_cost

        out["Proxy_LCOE_$/MWh"] = pd.to_numeric(out.apply(_proxy, axis=1), errors="coerce").fillna(0.0)

        # MAC ($/tCO2)
        def _mac(row):
            ghg = float(row.get("GHG_Reduction_tCO2_per_MWh", 0.0))
            lcoe = float(row.get("Proxy_LCOE_$/MWh", 0.0))
            if ghg <= 1e-9:
                return 0.0
            return lcoe / ghg

        out["MAC_$/tCO2"] = pd.to_numeric(out.apply(_mac, axis=1), errors="coerce").fillna(0.0)

        # ============================================================
        # Stack_MW for Supply Stack plot
        # - Default: Investment_MW
        # - Hydro: show peak dispatched MW instead (but keep Investment_MW = 0 in table)
        # ============================================================
        out["Stack_MW"] = out["Investment_MW"].copy()
        if "Imported Hydro" in disp.columns:
            hydro_peak_mw = float(pd.to_numeric(disp["Imported Hydro"], errors="coerce").fillna(0.0).max())
        else:
            hydro_peak_mw = 0.0
        out.loc[is_hydro, "Stack_MW"] = hydro_peak_mw

        # ============================================================
        # Optional debug: power balance residual
        # ============================================================
        if DEBUG:
            d = disp.copy()
            exclude = {"Hour", "Load_MW", "Load (gross)", "Load (net of T&D)"}
            cols = [c for c in d.columns if c not in exclude]
            LOAD_COL = "Load (net of T&D)" if "Load (net of T&D)" in d.columns else ("Load (gross)" if "Load (gross)" in d.columns else "Load_MW")
            net_supply = d[cols].sum(axis=1, numeric_only=True)
            resid = net_supply - pd.to_numeric(d[LOAD_COL], errors="coerce").fillna(0.0)

            _debug_print("\n=== BALANCE RESIDUAL CHECK (should be ~0) ===")
            _debug_print(
                pd.DataFrame({"Hour": d["Hour"], "Residual_MW": resid})
                .to_string(index=False, formatters={"Hour": "{:d}".format, "Residual_MW": fmt2})
            )
            _debug_print(f"Max abs residual: {fmt2(resid.abs().max())}")
            _debug_print("===========================================\n")

            _debug_print("\n=== DEBUG: FULL Investments + Proxy LCOE ===")
            debug_cols = out.columns.tolist()
            formatters = {c: fmt2 for c in debug_cols if pd.api.types.is_numeric_dtype(out[c])}
            _debug_print(out[debug_cols].to_string(index=False, formatters=formatters))
            _debug_print("===========================================\n")

        return out

    def _dispatch_totals_table():
        disp = model_results()["dispatch_df"].copy()
        disp["Hour"] = pd.to_numeric(disp["Hour"], errors="coerce").astype(int)
        disp = disp.sort_values("Hour").reset_index(drop=True)

        cols = [c for c in disp.columns if c != "Hour"]
        totals = disp[cols].sum(numeric_only=True)

        tbl = (
            totals.rename("Total_MWh")
            .reset_index()
            .rename(columns={"index": "Series"})
        )
        tbl["Abs_MWh"] = tbl["Total_MWh"].abs()
        tbl = tbl.sort_values("Abs_MWh", ascending=False).drop(columns=["Abs_MWh"])
        return tbl

    # ============================================================
    # Tables
    # ============================================================
    @output
    @render.table
    def summary_table():
        r = model_results()
        inv = _investment_with_cost_proxy()

        total_mw = float(inv["Investment_MW"].sum()) if "Investment_MW" in inv.columns else 0.0
        total_daily_capex = float(inv["Daily_Capex_$"].sum()) if "Daily_Capex_$" in inv.columns else 0.0
        total_upfront = float(inv["Upfront_Capex_$"].sum()) if "Upfront_Capex_$" in inv.columns else 0.0

        scenario_label = r["meta"]["demand_path"].replace("data/", "")
        if r["meta"]["scenario_name"]:
            scenario_label = f"{r['meta']['scenario_name']} ({scenario_label})"

        return (
            pd.DataFrame(
                {
                    "Metric": [
                        "Total investment (MW)",
                        "Total upfront capex ($)",
                        "Total daily capex ($/day)",
                        "Scenario",
                        "Max EV load (MW)",
                        "Discount rate",
                    ],
                    "Value": [
                        f"{total_mw:,.1f}",
                        f"{total_upfront:,.0f}",
                        f"{total_daily_capex:,.0f}",
                        scenario_label,
                        f"{float(r['meta']['max_ev_load_mw']):,.0f}",
                        f"{100.0 * float(r['meta']['discount_rate']):.2f}%",
                    ],
                }
            )
            .style
            .hide(axis="index")
            # body alignment
            .set_properties(subset=["Metric"], **{"text-align": "left"})
            .set_properties(subset=["Value"], **{"text-align": "center"})
            # header alignment
            .set_table_styles([
                {"selector": "th.col_heading.level0.col0", "props": [("text-align", "left")]},
                {"selector": "th.col_heading.level0.col1", "props": [("text-align", "center")]},
            ], overwrite=False)
        )

    @output
    @render.table
    def investment_table():
        inv = _investment_with_cost_proxy().copy()
        inv = inv[["Technology", "Investment_MW", "Proxy_LCOE_$/MWh"]].copy()

        for c in ["Investment_MW", "Proxy_LCOE_$/MWh"]:
            inv[c] = pd.to_numeric(inv[c], errors="coerce").fillna(0.0)

        return (
            inv.style
            .hide(axis="index")
            .format({
                "Investment_MW": "{:,.2f}",
                "Proxy_LCOE_$/MWh": "{:,.2f}",
            })
            # body alignment
            .set_properties(subset=["Technology"], **{"text-align": "left"})
            .set_properties(subset=["Investment_MW", "Proxy_LCOE_$/MWh"], **{"text-align": "right"})
            # headers left
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "left")]},
            ], overwrite=False)
        )

    @output
    @render.table
    def dispatch_totals():
        tbl = _dispatch_totals_table().copy()
        tbl["Total_MWh"] = tbl["Total_MWh"].map(lambda x: f"{float(x):,.2f}")
        return (
            tbl.style.hide(axis="index")
            # header left
            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}], overwrite=False)
            # body alignment
            .set_properties(subset=["Series"], **{"text-align": "left"})
            .set_properties(subset=["Total_MWh"], **{"text-align": "right"})
        )

    # ============================================================
    # Plots
    # ============================================================
    @output
    @render.plot
    def supply_stack():
        inv = _investment_with_cost_proxy().copy()

        inv["Stack_MW"] = pd.to_numeric(inv.get("Stack_MW", 0.0), errors="coerce").fillna(0.0)
        inv = inv[inv["Stack_MW"] > 1e-9].copy()

        if inv.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No investments selected.", ha="center", va="center")
            ax.axis("off")
            return fig

        inv = inv.sort_values("Proxy_LCOE_$/MWh").copy()
        inv["Block_Start"] = inv["Stack_MW"].cumsum() - inv["Stack_MW"]

        tech_list = inv["Technology"].tolist()
        color_map = _tech_color_map(tech_list)

        fig, ax = plt.subplots(figsize=(7, 4))
        for _, row in inv.iterrows():
            ax.bar(
                x=float(row["Block_Start"]),
                height=float(row["Proxy_LCOE_$/MWh"]),
                width=float(row["Stack_MW"]),
                color=color_map[row["Technology"]],
                align="edge",
                edgecolor="black",
                label=row["Technology"],
            )

        ax.set_title("Supply Stack: Proxy LCOE vs Cumulative Capacity")
        ax.set_xlabel("Cumulative Capacity (MW)")
        ax.set_ylabel("Proxy LCOE ($/MWh)")
        ax.grid(True, linestyle="--", alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        uniq = OrderedDict()
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h

        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=True,
            fontsize=8,
        )

        fig.tight_layout()
        return fig

    @output
    @render.plot
    def abatement_curve():
        inv = _investment_with_cost_proxy().copy()

        inv["Abatement_tCO2_per_day"] = pd.to_numeric(inv.get("Abatement_tCO2_per_day", 0.0), errors="coerce").fillna(0.0)
        inv["MAC_$/tCO2"] = pd.to_numeric(inv.get("MAC_$/tCO2", 0.0), errors="coerce").fillna(0.0)

        inv = inv[(inv["Abatement_tCO2_per_day"] > 1e-9) & (inv["MAC_$/tCO2"] > 0)].copy()

        if inv.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No abatement resources selected.", ha="center", va="center")
            ax.axis("off")
            return fig

        inv = inv.sort_values("MAC_$/tCO2").copy()
        inv["Block_Start"] = inv["Abatement_tCO2_per_day"].cumsum() - inv["Abatement_tCO2_per_day"]

        tech_list = inv["Technology"].tolist()
        color_map = _tech_color_map(tech_list)

        fig, ax = plt.subplots(figsize=(7, 4))
        for _, row in inv.iterrows():
            ax.bar(
                x=float(row["Block_Start"]),
                height=float(row["MAC_$/tCO2"]),
                width=float(row["Abatement_tCO2_per_day"]),
                color=color_map[row["Technology"]],
                align="edge",
                edgecolor="black",
                label=row["Technology"],
            )

        ax.set_title("Abatement Curve: MAC vs Cumulative Abatement")
        ax.set_xlabel("Cumulative abatement (tCO₂/day)")
        ax.set_ylabel("Marginal abatement cost ($/tCO₂)")
        ax.grid(True, linestyle="--", alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        uniq = OrderedDict()
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h

        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=True,
            fontsize=8,
        )

        fig.tight_layout()
        return fig

    @output
    @render.plot
    def dispatch_plot():
        d = model_results()["dispatch_df"].copy()
        req("Hour" in d.columns)

        positive_order = [
            "Solar PV", "Wind", "Net Metering", "Imported Hydro",
            "Battery Storage Discharge", "Pumped Storage Discharge",
            "Smart Grid Down",
            "T&D Savings",
        ]
        negative_order = [
            "EV Charging",
            "Battery Storage Charge", "Pumped Storage Charge",
            "Smart Grid Up",
        ]

        pos_cols = [c for c in positive_order if c in d.columns]
        neg_cols = [c for c in negative_order if c in d.columns]

        storage_base = {
            "Battery Storage": "#9D755D",
            "Pumped Storage": "#B279A2",
        }

        color_map = {
            "Solar PV": "#FDB813",
            "Wind": "#4C78A8",
            "Net Metering": "#72B7B2",
            "Imported Hydro": "#54A24B",
            "Smart Grid Down": "#FF9DA6",
            "Smart Grid Up": "#FF9DA6",
            "EV Charging": "#7F7F7F",
            "T&D Savings": "#E45756",
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        bottom_pos = np.zeros(len(d))
        for col in pos_cols:
            vals = pd.to_numeric(d[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            if col == "Battery Storage Discharge":
                c = storage_base["Battery Storage"]
            elif col == "Pumped Storage Discharge":
                c = storage_base["Pumped Storage"]
            else:
                c = color_map.get(col, "#999999")

            ax.bar(
                d["Hour"], vals, bottom=bottom_pos,
                label=col, color=c,
                edgecolor="white", linewidth=0.3,
                width=1.0, align="center"
            )
            bottom_pos += vals

        bottom_neg = np.zeros(len(d))
        for col in neg_cols:
            vals = pd.to_numeric(d[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            if col == "Battery Storage Charge":
                c = storage_base["Battery Storage"]
                ax.bar(
                    d["Hour"], vals, bottom=bottom_neg,
                    label=col, color=c, hatch="///",
                    edgecolor="black", linewidth=0.3,
                    width=1.0, align="center"
                )
            elif col == "Pumped Storage Charge":
                c = storage_base["Pumped Storage"]
                ax.bar(
                    d["Hour"], vals, bottom=bottom_neg,
                    label=col, color=c, hatch="///",
                    edgecolor="black", linewidth=0.3,
                    width=1.0, align="center"
                )
            elif col == "Smart Grid Up":
                ax.bar(
                    d["Hour"], vals, bottom=bottom_neg,
                    label=col, color=color_map.get(col, "#999999"), hatch="///",
                    edgecolor="black", linewidth=0.3,
                    width=1.0, align="center"
                )
            else:
                ax.bar(
                    d["Hour"], vals, bottom=bottom_neg,
                    label=col, color=color_map.get(col, "#999999"),
                    edgecolor="white", linewidth=0.3,
                    width=1.0, align="center"
                )

            bottom_neg += vals

        # ---- Step-function load lines aligned to bar edges ----
        x_edges = np.arange(-0.5, 24.0, 1.0)  # -0.5 ... 23.5 (25 points)

        def plot_step_load(col_name: str, label: str, linestyle: str):
            if col_name not in d.columns:
                return
            y = pd.to_numeric(d[col_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            y_step = np.r_[y, y[-1]]
            ax.step(
                x_edges, y_step, where="post",
                linestyle=linestyle, linewidth=2.2, color="black",
                label=label, zorder=10
            )

        plot_step_load("Load (gross)", "Load (gross)", "--")
        plot_step_load("Load (net of T&D)", "Load (net of T&D)", ":")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Hourly Dispatch (Positive = Supply/Discharge/DSM-Down, Negative = EV/Charging/DSM-Up)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")

        ax.set_xticks(np.arange(0, 24, 1))
        ax.set_xlim(-0.5, 23.5)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        h2, l2 = [], []
        for h, lab in zip(handles, labels):
            if lab and lab not in seen:
                h2.append(h)
                l2.append(lab)
                seen.add(lab)

        fig.subplots_adjust(bottom=0.33)
        fig.legend(
            h2, l2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.16, 1, 1])
        return fig


app = App(app_ui, server)
