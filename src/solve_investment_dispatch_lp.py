"""
Investment + Dispatch Optimization Solver

This module implements a linear programming model that:
- Determines optimal investment capacity by technology
- Dispatches resources hourly to meet load
- Models storage and smart grid as balancing resources
- Applies renewable availability constraints
- Computes proxy LCOE and abatement metrics

The solver is designed to be called from a Shiny application
but can also be used standalone for batch analysis.

Author: Anthony Peluso
Project: Investment Optimization Dashboard
"""

# ============================
# Third-party imports
# ============================
import pandas as pd
import numpy as np
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    value,
)

# -----------------------------
# helpers
# -----------------------------
def _norm(name: str) -> str:
    """Normalize tech names across CSVs: spaces -> underscores, trim."""
    return str(name).strip().replace(" ", "_")

def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# ============================================================
# Investment + Dispatch LP
# ============================================================
def solve_investment_dispatch_lp(
    tech_df: pd.DataFrame,
    load_df: pd.DataFrame,
    ev_profile: pd.DataFrame,
    availability_df: pd.DataFrame,
    max_ev_load_mw: float,
    discount_rate: float,
    *,
    td_full_savings_frac: float = 0.10,          # e.g. 10% of gross load at "full build"
    hydro_backstop_limit_mw: float = 10000.0,    # max dispatch per hour
    lifetime_years: int = 20,
):
    """
    Clean reference formulation (1-hour slices):

    SUPPLY:
      - All supply gen limited by cap[tech] * availability[tech,h]
      - Net Metering tied to Solar availability (behind-the-meter PV)

    T&D:
      - Not dispatched as generation.
      - Reduces net load proportionally to gross load:
          savings[h] = (cap_TD / TD_cap_limit) * td_full_savings_frac * gross_load[h]
      - Report:
          Load (gross), Load (net of T&D), T&D Savings

    FLEX (Balancing / Smart Grid):
      - Battery, Pumped Storage: charge/discharge + SOC, eta=0.9 on discharge
      - Smart Grid: modeled as balancing (Up/Down), eta=1.0
      - SOC start/end = 0, SOC <= cap (cap is energy capacity since dt=1h)
      - Optional daily throughput budget for Demand-side based on Load_Reduction_MW * 24

    HYDRO:
      - Imported Hydro investment fixed at 0 (cap upBound = 0)
      - Dispatch allowed up to hydro_backstop_limit_mw (backstop)
    """

    # -----------------------------
    # basic time index
    # -----------------------------
    hours = load_df["Hour"].astype(int).tolist()
    H = len(hours)

    gross_load = dict(zip(load_df["Hour"].astype(int), load_df["Load_MW"]))
    ev_load = {
        h: float(max_ev_load_mw)
        * float(ev_profile.loc[ev_profile["Hour"].astype(int) == h, "EV_Profile"].iloc[0])
        for h in hours
    }

    # -----------------------------
    # normalize discount rate
    # -----------------------------
    r = float(discount_rate)
    if r > 1.0:
        r = r / 100.0

    if r <= 0:
        crf = 1.0 / float(lifetime_years)
    else:
        crf = (r * (1 + r) ** lifetime_years) / ((1 + r) ** lifetime_years - 1)
    capex_daily_factor = crf / 365.0

    # -----------------------------
    # availability lookup
    # -----------------------------
    # Expect availability_df: Hour, Technology, Availability
    avail = {
        (_norm(row["Technology"]), int(row["Hour"])): float(row["Availability"])
        for _, row in availability_df.iterrows()
    }

    def get_avail(tech_name: str, h: int) -> float:
        """
        Returns availability for a technology at hour h.
        Default 1.0 if missing.

        Special rule: Net Metering uses Solar availability.
        """
        tkey = _norm(tech_name)

        if tkey in ("Net_Metering", "NetMetering", "Net_Metering_PV"):
            # force behind-the-meter solar shape
            solar_key = "Solar_PV"
            return float(avail.get((solar_key, int(h)), 1.0))

        return float(avail.get((tkey, int(h)), 1.0))

    # -----------------------------
    # identify technology buckets
    # -----------------------------
    tech_df = tech_df.copy()
    tech_df["Technology"] = tech_df["Technology"].astype(str)

    # Special names (match your CSV labels)
    TD_NAME = "T&D Upgrades"
    HYDRO_NAME = "Imported Hydro"
    SMART_NAME = "Smart Grid"

    # helper flags
    tech_df["DSM_Mode"] = tech_df["DSM_Mode"] if "DSM_Mode" in tech_df.columns else "0"

    # Supply generation technologies (exclude T&D, exclude hydro-as-backstop)
    supply_techs = tech_df.query(
        "Dispatchability in ['Non-dispatchable','Dispatchable']"
    ).copy()
    supply_techs = supply_techs[~supply_techs["Technology"].isin([TD_NAME, HYDRO_NAME])].copy()

    # Balancing storage
    storage_techs = tech_df.query("Dispatchability == 'Balancing'").copy()

    # Smart Grid as balancing DSM (if present)
    smart_grid_techs = tech_df[tech_df["Technology"] == SMART_NAME].copy()

    # T&D tech row (if present)
    td_row = tech_df[tech_df["Technology"] == TD_NAME].copy()

    # Hydro row (if present)
    hydro_row = tech_df[tech_df["Technology"] == HYDRO_NAME].copy()

    # Flex techs that will get charge/discharge/SOC:
    # - storage techs (Balancing)
    # - smart grid (balancing-like)
    flex_techs = pd.concat([storage_techs, smart_grid_techs], axis=0)
    flex_techs = flex_techs.drop_duplicates(subset=["Technology"]).copy()

    # efficiency on discharge: storage 0.9, smart grid 1.0
    eta = {}
    for _, t in flex_techs.iterrows():
        if t["Dispatchability"] == "Balancing":
            eta[t["Technology"]] = 0.90
        else:
            eta[t["Technology"]] = 1.00
    inv_eta = {k: (1.0 / v) for k, v in eta.items()}  # constant multipliers

    # -----------------------------
    # LP
    # -----------------------------
    prob = LpProblem("Investment_Dispatch", LpMinimize)

    # capacity decision vars (MW == MWh due to 1h slices) for all techs
    cap = {
        t["Technology"]: LpVariable(
            f"Cap_{_norm(t['Technology'])}",
            lowBound=0,
            upBound=_as_float(t.get("Capacity_Limit_MW", 0.0), 0.0)
        )
        for _, t in tech_df.iterrows()
    }

    # Force hydro investment to 0
    if HYDRO_NAME in cap:
        cap[HYDRO_NAME].upBound = 0.0

    # Supply generation vars
    gen = {
        (t["Technology"], h): LpVariable(f"G_{_norm(t['Technology'])}_{h}", lowBound=0)
        for _, t in supply_techs.iterrows()
        for h in hours
    }

    # Hydro dispatch var (dispatch-only backstop)
    hydro_gen = {
        h: LpVariable(f"G_{_norm(HYDRO_NAME)}_{h}", lowBound=0)
        for h in hours
    } if HYDRO_NAME in tech_df["Technology"].values else {}

    # Flex vars: charge/discharge and SOC
    charge = {
        (t["Technology"], h): LpVariable(f"Ch_{_norm(t['Technology'])}_{h}", lowBound=0)
        for _, t in flex_techs.iterrows()
        for h in hours
    }
    discharge = {
        (t["Technology"], h): LpVariable(f"Dis_{_norm(t['Technology'])}_{h}", lowBound=0)
        for _, t in flex_techs.iterrows()
        for h in hours
    }
    # SOC indexed by step i = 0..H
    soc = {
        (t["Technology"], i): LpVariable(f"SOC_{_norm(t['Technology'])}_{i}", lowBound=0)
        for _, t in flex_techs.iterrows()
        for i in range(H + 1)
    }

    # -----------------------------
    # T&D savings definition
    # -----------------------------
    # savings[h] = scale * td_full_savings_frac * gross_load[h]
    # where scale = cap_TD / TD_cap_limit (0..1)
    td_savings = {h: 0.0 for h in hours}
    td_scale_var = None

    if not td_row.empty:
        td_cap_limit = float(td_row["Capacity_Limit_MW"].iloc[0]) if "Capacity_Limit_MW" in td_row.columns else 0.0
        if td_cap_limit > 0 and TD_NAME in cap:
            td_scale_var = LpVariable("TD_Scale", lowBound=0, upBound=1)
            # tie scale to investment: cap_TD = scale * limit
            prob += cap[TD_NAME] == td_scale_var * td_cap_limit, "TD_scale_link"
        else:
            # if limit missing or zero, treat as no effect
            td_scale_var = None

    # -----------------------------
    # Hourly balance constraints
    # -----------------------------
    for h in hours:
        gross = float(gross_load[h])
        # savings at hour h is an expression if td_scale_var exists, else 0
        if td_scale_var is not None:
            savings_expr = td_scale_var * td_full_savings_frac * gross
        else:
            savings_expr = 0.0

        net_load_expr = gross - savings_expr

        prob += (
            # Supply generation
            lpSum(gen[(t["Technology"], h)] for _, t in supply_techs.iterrows())
            # Hydro backstop
            + (hydro_gen[h] if h in hydro_gen else 0.0)
            # Flex discharge reduces need
            + lpSum(discharge[(t["Technology"], h)] for _, t in flex_techs.iterrows())
            ==
            # Net load after T&D
            net_load_expr
            # EV adds load
            + float(ev_load[h])
            # Flex charging adds load
            + lpSum(charge[(t["Technology"], h)] for _, t in flex_techs.iterrows())
        ), f"PowerBalance_{h}"

    # -----------------------------
    # Supply availability limits
    # -----------------------------
    for _, t in supply_techs.iterrows():
        tech = t["Technology"]
        for h in hours:
            a = get_avail(tech, h)
            prob += gen[(tech, h)] <= cap[tech] * a, f"Avail_{_norm(tech)}_{h}"

    # Hydro dispatch limits
    if hydro_gen:
        for h in hours:
            prob += hydro_gen[h] <= float(hydro_backstop_limit_mw), f"HydroBackstop_{h}"

    # -----------------------------
    # Flex constraints (storage + smart grid)
    # -----------------------------
    for _, t in flex_techs.iterrows():
        tech = t["Technology"]

        # SOC start/end
        prob += soc[(tech, 0)] == 0, f"SOC_start_{_norm(tech)}"
        prob += soc[(tech, H)] == 0, f"SOC_end_{_norm(tech)}"

        # SOC dynamics + bounds
        for i, h in enumerate(hours):
            prob += (
                soc[(tech, i + 1)]
                ==
                soc[(tech, i)]
                + charge[(tech, h)]
                - discharge[(tech, h)] * inv_eta[tech]   # linear (mult by constant)
            ), f"SOC_dyn_{_norm(tech)}_{h}"

            # bounds
            prob += soc[(tech, i)] <= cap[tech], f"SOC_cap_{_norm(tech)}_{h}"
            prob += charge[(tech, h)] <= cap[tech], f"Ch_cap_{_norm(tech)}_{h}"
            prob += discharge[(tech, h)] <= cap[tech], f"Dis_cap_{_norm(tech)}_{h}"

        # optional daily throughput cap for Smart Grid using Load_Reduction_MW
        if tech == SMART_NAME and "Load_Reduction_MW" in tech_df.columns:
            lr = float(tech_df.loc[tech_df["Technology"] == tech, "Load_Reduction_MW"].iloc[0])
            if lr > 0:
                daily_budget = lr * 24.0
                prob += lpSum(charge[(tech, h)] for h in hours) <= daily_budget, f"SG_daily_budget_ch_{_norm(tech)}"
                prob += lpSum(discharge[(tech, h)] for h in hours) <= daily_budget, f"SG_daily_budget_dis_{_norm(tech)}"

    # -----------------------------
    # Objective: daily capex + variable cost on supply + hydro
    # -----------------------------
    capex_cost = lpSum(
        cap[t["Technology"]] * float(t["Capex_$/kW"]) * 1000.0 * capex_daily_factor
        for _, t in tech_df.iterrows()
        # (hydro cap is fixed 0 anyway; safe)
    )

    # variable costs:
    # - supply techs use Var_Cost_$/MWh
    # - hydro uses Var_Cost_$/MWh if present in tech_df, else 0
    var_cost_supply = lpSum(
        gen[(t["Technology"], h)] * float(t["Var_Cost_$/MWh"])
        for _, t in supply_techs.iterrows()
        for h in hours
    )

    hydro_var = 0.0
    if hydro_gen:
        if not hydro_row.empty and "Var_Cost_$/MWh" in hydro_row.columns:
            hydro_v = float(hydro_row["Var_Cost_$/MWh"].iloc[0])
        else:
            hydro_v = 0.0
        hydro_var = lpSum(hydro_gen[h] * hydro_v for h in hours)

    prob += capex_cost + var_cost_supply + hydro_var

    # -----------------------------
    # Solve
    # -----------------------------
    prob.solve()
    print("DEBUG Status:", LpStatus[prob.status])
    print("DEBUG Objective:", value(prob.objective))

    # -----------------------------
    # Build dispatch_df (matches your UI stacking convention)
    # -----------------------------
    # Convention:
    #   Positive = supply/discharge + "Down" type effects (incl. T&D savings)
    #   Negative = EV + charging + "Up" type effects
    # Here:
    #   - Storage charge is negative, discharge positive
    #   - Smart Grid Down = discharge (positive), Smart Grid Up = charge (negative)
    #   - EV is negative
    #   - T&D Savings is positive
    rows = []
    balance_residual = []

    # Compute realized T&D savings for output (post-solve)
    td_cap_limit = None
    if not td_row.empty and "Capacity_Limit_MW" in td_row.columns:
        td_cap_limit = float(td_row["Capacity_Limit_MW"].iloc[0])

    td_cap_val = float(cap[TD_NAME].varValue or 0.0) if (TD_NAME in cap) else 0.0
    td_scale_val = (td_cap_val / td_cap_limit) if (td_cap_limit and td_cap_limit > 0) else 0.0

    for h in hours:
        gross = float(gross_load[h])
        savings = td_scale_val * td_full_savings_frac * gross
        net = gross - savings

        r = {
            "Hour": int(h),
            "Load (gross)": gross,
            "Load (net of T&D)": net,
            "T&D Savings": savings,
            "EV Charging": -float(ev_load[h]),
        }

        # supply techs
        for _, t in supply_techs.iterrows():
            tech = t["Technology"]
            r[tech] = float(gen[(tech, h)].varValue or 0.0)

        # hydro dispatch
        if hydro_gen:
            r[HYDRO_NAME] = float(hydro_gen[h].varValue or 0.0)

        # storage techs (explicit names)
        for _, t in storage_techs.iterrows():
            tech = t["Technology"]
            r[f"{tech} Discharge"] = float(discharge[(tech, h)].varValue or 0.0)
            r[f"{tech} Charge"] = -float(charge[(tech, h)].varValue or 0.0)

        # smart grid as balancing up/down
        if not smart_grid_techs.empty:
            tech = SMART_NAME
            r[f"{tech} Down"] = float(discharge[(tech, h)].varValue or 0.0)
            r[f"{tech} Up"] = -float(charge[(tech, h)].varValue or 0.0)

        # residual check (should be ~0)
        lhs = 0.0
        rhs = 0.0

        # lhs: supply + hydro + discharges
        lhs += sum(float(r.get(t["Technology"], 0.0)) for _, t in supply_techs.iterrows())
        if hydro_gen:
            lhs += float(r.get(HYDRO_NAME, 0.0))
        lhs += sum(float(r.get(f"{t['Technology']} Discharge", 0.0)) for _, t in storage_techs.iterrows())
        if not smart_grid_techs.empty:
            lhs += float(r.get(f"{SMART_NAME} Down", 0.0))

        # rhs: net load + EV + charges
        rhs += net
        rhs += float(ev_load[h])
        rhs += sum(-float(r.get(f"{t['Technology']} Charge", 0.0)) for _, t in storage_techs.iterrows())  # remove neg sign
        if not smart_grid_techs.empty:
            rhs += -float(r.get(f"{SMART_NAME} Up", 0.0))  # remove neg sign

        balance_residual.append((h, lhs - rhs))

        rows.append(r)

    dispatch_df = pd.DataFrame(rows)

    # -----------------------------
    # Investment output (NO Daily_Capex column)
    # -----------------------------
    inv_rows = []
    for _, t in tech_df.iterrows():
        tech = t["Technology"]
        mw = float(cap[tech].varValue or 0.0) if tech in cap else 0.0

        # Force hydro investment to 0 in output
        if tech == HYDRO_NAME:
            mw = 0.0

        inv_rows.append({
            "Technology": tech,
            "Investment_MW": mw,
            "Upfront_Capex_$": mw * float(t["Capex_$/kW"]) * 1000.0,
        })
    investment_df = pd.DataFrame(inv_rows)

    # -----------------------------
    # Diagnostics: totals + residual check
    # -----------------------------
    print("\n=== TOTAL DISPATCH BY COLUMN (Daily MWh) ===")
    cols = [c for c in dispatch_df.columns if c != "Hour"]
    totals = dispatch_df[cols].sum(numeric_only=True)

    summary_tbl = (
        totals.rename("Total_MWh")
        .reset_index()
        .rename(columns={"index": "Series"})
    )
    summary_tbl["Abs_MWh"] = summary_tbl["Total_MWh"].abs()
    summary_tbl = summary_tbl.sort_values("Abs_MWh", ascending=False).drop(columns=["Abs_MWh"])

    pd.set_option("display.max_rows", 250)
    pd.set_option("display.width", 160)
    print(summary_tbl.to_string(index=False, formatters={"Total_MWh": "{:,.2f}".format}))
    print("===========================================\n")

    # residual report
    res_df = pd.DataFrame(balance_residual, columns=["Hour", "Residual_MW"])
    worst = float(res_df["Residual_MW"].abs().max()) if not res_df.empty else 0.0
    print("=== BALANCE RESIDUAL CHECK (should be ~0) ===")
    print(res_df.to_string(index=False, formatters={"Residual_MW": "{:,.6f}".format}))
    print("Max abs residual:", f"{worst:,.6f}")
    print("===========================================\n")

    # availability coverage check (helps catch name mismatches)
    print("=== AVAILABILITY COVERAGE CHECK ===")
    for tech in supply_techs["Technology"].unique():
        missing = [h for h in hours if (_norm(tech), int(h)) not in avail]
        # note: Net Metering is special-cased to solar, so we don't treat it as "missing"
        if _norm(tech) == "Net_Metering":
            print("INFO: Net Metering availability tied to Solar PV (forced).")
            continue
        if missing:
            print(f"WARNING: {tech} missing availability for {len(missing)}/24 hours. Default=1.0 for missing.")
        else:
            print(f"OK: {tech} has availability for all 24 hours.")
    print("===================================\n")

    return investment_df, dispatch_df
