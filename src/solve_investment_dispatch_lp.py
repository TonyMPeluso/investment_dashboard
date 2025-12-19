import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.strip('"').str.strip("'")
    return df


def _capital_recovery_factor(discount_rate: float, asset_life_years: float) -> float:
    """
    CRF converts upfront capex into an equivalent annual payment.
    """
    r = float(discount_rate)
    n = float(asset_life_years)

    if n <= 0:
        return 1.0
    if abs(r) < 1e-12:
        return 1.0 / n

    x = (1.0 + r) ** n
    return (r * x) / (x - 1.0)


def solve_investment_dispatch_lp(
    tech_params_path: str,
    load_curve_file: str,
    dispatch_profiles_path: str,
    ev_profile_path: str,
    ev_max_mw: float = 0.0,
    hydro_import_cap_mw: float = 5000.0,          # UI value: max imports (MW) each hour]
    crf = None,
    import_penalty_per_mwh: float = 200.0,        # autonomy penalty added ONLY in optimization objective
    hours_of_storage: float = 4.0,
    peak_hours: Tuple[int, int] = (16, 22),       # DSM allowed in [start, end) -> 16..21
    discount_rate: float = 0.07,                  # for CRF
    asset_life_years: float = 25.0,               # for CRF
    capex_daily_divisor: float = 365.0,           # annual -> daily scaling
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Joint capacity + dispatch optimization (single LP), with Backstop Imports.

    Key behavior:
      - NO "unserved" slack variable exists.
      - Any shortfall must be met by backstop imports, capped by hydro_import_cap_mw each hour.

    Decision variables:
      - Cap_MW[tech] for local Supply/DSM/Storage power capacity
      - Gen_MW[tech,h] for local supply dispatch
      - DSM_MW[tech,h] for demand-side reductions (treated as MWh saved)
      - Charge/Discharge/SOC for storage
      - Import_MW[h] for backstop imports, 0 <= Import_MW[h] <= hydro_import_cap_mw

    Objective (daily):
      min  Sum_tech (Capex_$/kW * 1000 * Cap_MW * (CRF/365))
         + Sum_{tech,h} (Var_Cost_$/MWh * dispatch_MWh)
         + Sum_h ( (Import_Price + import_penalty_per_mwh) * Import_MW[h] )

    Notes:
      - Imported Hydro is treated as a backstop purchase, not an "investment".
      - Import_Price is read from tech_parameters.csv Var_Cost_$/MWh for "Imported Hydro".
      - If the model cannot meet load within the import cap, it will become infeasible (as intended).
    """

    # ----------------------------
    # Robust paths
    # ----------------------------
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, ".."))

    tech_path = os.path.abspath(os.path.join(project_root, tech_params_path))
    load_path = os.path.abspath(os.path.join(project_root, load_curve_file))
    prof_path = os.path.abspath(os.path.join(project_root, dispatch_profiles_path))
    evp_path = os.path.abspath(os.path.join(project_root, ev_profile_path))

    tech = _clean_cols(pd.read_csv(tech_path))
    load = _clean_cols(pd.read_csv(load_path))
    prof = _clean_cols(pd.read_csv(prof_path))
    evp = _clean_cols(pd.read_csv(evp_path))

    # ----------------------------
    # Input checks / normalization
    # ----------------------------
    if "Technology" not in tech.columns:
        raise ValueError("tech_parameters.csv must include a 'Technology' column")

    if "Load_MW" not in load.columns:
        raise ValueError(f"Load file must include Load_MW. Found: {list(load.columns)}")
    if "Hour" not in load.columns:
        load["Hour"] = range(len(load))

    if "Hour" not in evp.columns or "EV_Profile" not in evp.columns:
        raise ValueError(f"EV profile must include Hour and EV_Profile. Found: {list(evp.columns)}")

    # Ensure expected tech columns exist
    expected_cols = [
        "Capex_$/kW",
        "Var_Cost_$/MWh",
        "GHG_Reduction_tCO2_per_MWh",
        "Reliability_Score",
        "Capacity_Limit_MW",
        "Load_Reduction_MW",
        "Dispatchability",
    ]
    for c in expected_cols:
        if c not in tech.columns:
            tech[c] = 0.0

    tech["Technology"] = tech["Technology"].astype(str).str.strip()
    tech["Dispatchability"] = tech["Dispatchability"].astype(str).str.strip()

    for c in [
        "Capex_$/kW",
        "Var_Cost_$/MWh",
        "Capacity_Limit_MW",
        "Load_Reduction_MW",
        "GHG_Reduction_tCO2_per_MWh",
        "Reliability_Score",
    ]:
        tech[c] = pd.to_numeric(tech[c], errors="coerce").fillna(0.0)

    hours: List[int] = list(load["Hour"])
    demand = load["Load_MW"].to_numpy(dtype=float)

    ev_max_mw = float(ev_max_mw)
    evp = evp.set_index("Hour").reindex(hours).fillna(0.0)
    ev_profile = evp["EV_Profile"].to_numpy(dtype=float)

    ev_mw = ev_max_mw * ev_profile
    net_load = demand + ev_mw

    # ----------------------------
    # Tech maps + sets
    # ----------------------------
    disp = tech.set_index("Technology")["Dispatchability"].to_dict()
    capex_kw = tech.set_index("Technology")["Capex_$/kW"].to_dict()
    var_cost = tech.set_index("Technology")["Var_Cost_$/MWh"].to_dict()
    cap_limit = tech.set_index("Technology")["Capacity_Limit_MW"].to_dict()
    load_red_limit = tech.set_index("Technology")["Load_Reduction_MW"].to_dict()
    ghg_rate = tech.set_index("Technology")["GHG_Reduction_tCO2_per_MWh"].to_dict()
    rel_score = tech.set_index("Technology")["Reliability_Score"].to_dict()

    all_techs = list(tech["Technology"].unique())

    # Local supply techs EXCLUDING Imported Hydro (imports handled separately)
    local_supply_techs = [
        t for t in all_techs
        if disp.get(t) in ["Non-dispatchable", "Dispatchable"] and t != "Imported Hydro"
    ]
    dsm_techs = [t for t in all_techs if disp.get(t) == "Demand-side"]
    storage_techs = [t for t in all_techs if disp.get(t) == "Balancing"]

    import_price = float(var_cost.get("Imported Hydro", 0.0))  # "normal hydro price" for reporting

    if verbose:
        print("DEBUG local_supply_techs:", local_supply_techs)
        print("DEBUG dsm_techs:", dsm_techs)
        print("DEBUG storage_techs:", storage_techs)
        print("DEBUG import_price ($/MWh):", import_price)
        print("DEBUG import_penalty_per_mwh ($/MWh):", float(import_penalty_per_mwh))
        print("DEBUG hydro_import_cap_mw (MW):", float(hydro_import_cap_mw))

    # ----------------------------
    # Availability profiles
    # ----------------------------
    availability_mapping: Dict[str, str] = {
        "Solar PV": "Solar PV",
        "Wind": "Wind",
        # Treat net metering like solar by default if you want it tied to solar profile:
        "Net Metering": "Solar PV",
    }

    def availability_factor(tech_name: str, hour: int) -> float:
        col = availability_mapping.get(tech_name)
        if not col or col not in prof.columns:
            return 1.0
        if "Hour" not in prof.columns:
            return 1.0
        row = prof.loc[prof["Hour"] == hour, col]
        return 1.0 if row.empty else float(row.iloc[0])

    # ----------------------------
    # Daily capex factor (Option 1)
    # ----------------------------
    crf = _capital_recovery_factor(discount_rate, asset_life_years)
    capex_daily_factor = float(crf) / float(capex_daily_divisor)

    if verbose:
        print("DEBUG CRF:", float(crf), "capex_daily_factor:", float(capex_daily_factor))

    # ----------------------------
    # Build model
    # ----------------------------
    model = LpProblem("Investment_and_Dispatch", LpMinimize)

    def vname(prefix: str, tech_name: str, h: Optional[int] = None) -> str:
        base = tech_name.replace(" ", "_").replace("-", "_")
        return f"{prefix}_{base}" if h is None else f"{prefix}_{base}_{h}"

    # Capacity decision variables (MW)
    cap_mw: Dict[str, LpVariable] = {}

    # Local supply capacities
    for t in local_supply_techs:
        cap_mw[t] = LpVariable(
            vname("CapMW", t),
            lowBound=0,
            upBound=float(cap_limit.get(t, 0.0)),
        )

    # DSM capacities (use Load_Reduction_MW if present, else fallback to capacity limit)
    for t in dsm_techs:
        ub = float(load_red_limit.get(t, 0.0))
        if ub <= 0:
            ub = float(cap_limit.get(t, 0.0))
        cap_mw[t] = LpVariable(vname("CapMW", t), lowBound=0, upBound=ub)

    # Storage power capacities
    for s in storage_techs:
        cap_mw[s] = LpVariable(
            vname("CapMW", s),
            lowBound=0,
            upBound=float(cap_limit.get(s, 0.0)),
        )

    # Dispatch variables (MW == MWh over 1 hour)
    gen = {(t, h): LpVariable(vname("Gen", t, h), lowBound=0) for t in local_supply_techs for h in hours}
    dsm = {(t, h): LpVariable(vname("DSM", t, h), lowBound=0) for t in dsm_techs for h in hours}

    charge = {(s, h): LpVariable(vname("Chg", s, h), lowBound=0) for s in storage_techs for h in hours}
    discharge = {(s, h): LpVariable(vname("Dis", s, h), lowBound=0) for s in storage_techs for h in hours}
    soc = {(s, h): LpVariable(vname("SOC", s, h), lowBound=0) for s in storage_techs for h in hours}

    # Backstop imports (Imported Hydro), capped by UI each hour
    imports = {
        h: LpVariable(vname("Import", "Hydro", h), lowBound=0, upBound=float(hydro_import_cap_mw))
        for h in hours
    }

    # ----------------------------
    # Constraints
    # ----------------------------

    # Local supply availability/capacity
    for t in local_supply_techs:
        for h in hours:
            model += gen[(t, h)] <= cap_mw[t] * availability_factor(t, h)

    # DSM + peak-only restriction
    peak_start, peak_end = int(peak_hours[0]), int(peak_hours[1])
    peak_set = set(range(peak_start, peak_end))

    for t in dsm_techs:
        for h in hours:
            model += dsm[(t, h)] <= cap_mw[t]
            if h not in peak_set:
                model += dsm[(t, h)] == 0

    # Storage (SOC start/end = 0, SOC bounds from power * hours_of_storage)
    eta_c, eta_d = 0.95, 0.95

    for s in storage_techs:
        power_cap = cap_mw[s]
        energy_cap = power_cap * float(hours_of_storage)

        for h in hours:
            model += charge[(s, h)] <= power_cap
            model += discharge[(s, h)] <= power_cap
            model += soc[(s, h)] <= energy_cap

        # Pin start/end
        model += soc[(s, hours[0])] == 0
        model += soc[(s, hours[-1])] == 0

        # SOC dynamics for ALL hours (including hour 0, wrapping from last hour)
        for idx, h in enumerate(hours):
            prev_h = hours[idx - 1]  # wraps: idx=0 -> prev_h = last hour
            model += (
                soc[(s, h)]
                == soc[(s, prev_h)]
                + eta_c * charge[(s, h)]
                - (1.0 / eta_d) * discharge[(s, h)]
            )


    # Load balance: imports cover remaining shortfall (within cap)
    for idx, h in enumerate(hours):
        supply_terms = lpSum(gen[(t, h)] for t in local_supply_techs)
        dsm_terms = lpSum(dsm[(t, h)] for t in dsm_techs)
        dis_terms = lpSum(discharge[(s, h)] for s in storage_techs)
        chg_terms = lpSum(charge[(s, h)] for s in storage_techs)

        model += (
            supply_terms + dsm_terms + dis_terms - chg_terms + imports[h]
            == float(net_load[idx])
        )

    # ----------------------------
    # Objective (Option 1)
    # ----------------------------
    capex_term = lpSum(
        float(capex_kw.get(t, 0.0)) * 1000.0 * cap_mw[t] * capex_daily_factor
        for t in cap_mw.keys()
    )

    var_term = lpSum(
        gen[(t, h)] * float(var_cost.get(t, 0.0))
        for t in local_supply_techs
        for h in hours
    )

    var_term += lpSum(
        dsm[(t, h)] * float(var_cost.get(t, 0.0))
        for t in dsm_techs
        for h in hours
    )

    # Storage variable cost applied to discharge MWh (your preference)
    var_term += lpSum(
        discharge[(s, h)] * float(var_cost.get(s, 0.0))
        for s in storage_techs
        for h in hours
    )

    # Imports: normal price + penalty (penalty enforces autonomy in optimization only)
    imports_term = lpSum(
        imports[h] * float(import_price + float(import_penalty_per_mwh))
        for h in hours
    )

    model += capex_term + var_term + imports_term

    # ----------------------------
    # Solve
    # ----------------------------
    model.solve()
    if LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"Investment+Dispatch LP not optimal. Status={LpStatus[model.status]}")

    # ----------------------------
    # Build outputs
    # ----------------------------

    # Investment summary
    cap_rows = []
    for t in all_techs:
        dtype = disp.get(t, "")

        # Imported Hydro is not an invested asset (it's an import option)
        inv_mw = 0.0
        if t in cap_mw:
            inv_mw = float(cap_mw[t].varValue or 0.0)

        cap_rows.append(
            {
                "Technology": t,
                "Dispatchability": dtype,
                "Investment_MW": float(inv_mw),
                "Capex_$/kW": float(capex_kw.get(t, 0.0)),
                "Var_Cost_$/MWh": float(var_cost.get(t, 0.0)),
                "GHG_Reduction_tCO2_per_MWh": float(ghg_rate.get(t, 0.0)),
                "Reliability_Score": float(rel_score.get(t, 0.0)),
            }
        )
    inv_df = pd.DataFrame(cap_rows)

    # Dispatch time series
    rows = []
    for idx, h in enumerate(hours):
        row = {
            "Hour": int(h),
            "Load_MW": float(demand[idx]),
            "EV_MW": float(ev_mw[idx]),
            "Net_Load_MW": float(net_load[idx]),
            # expose imports as Imported Hydro in the time series
            "Imported Hydro": float(imports[h].varValue or 0.0),
        }

        for t in local_supply_techs:
            row[t] = float(gen[(t, h)].varValue or 0.0)
        for t in dsm_techs:
            row[t] = float(dsm[(t, h)].varValue or 0.0)
        for s in storage_techs:
            row[f"{s} Charge"] = float(charge[(s, h)].varValue or 0.0)
            row[f"{s} Discharge"] = float(discharge[(s, h)].varValue or 0.0)
            row[f"{s} SOC"] = float(soc[(s, h)].varValue or 0.0)

        rows.append(row)

    disp_df = pd.DataFrame(rows)

    # ----------------------------
    # Terminal totals (Daily MWh)
    # ----------------------------
    if verbose:
        supply_cols = [c for c in ["Solar PV", "Wind", "Net Metering", "T&D Upgrades", "Imported Hydro"] if c in disp_df.columns]
        dsm_cols = [c for c in ["Time-of-Day Pricing", "Smart Grid"] if c in disp_df.columns]
        storage_dis_cols = [c for c in ["Battery Storage Discharge", "Pumped Storage Discharge"] if c in disp_df.columns]
        storage_chg_cols = [c for c in ["Battery Storage Charge", "Pumped Storage Charge"] if c in disp_df.columns]
        ev_col = "EV_MW" if "EV_MW" in disp_df.columns else None

        totals = {
            "Total Generation (MWh)": float(disp_df[supply_cols].sum().sum()) if supply_cols else 0.0,
            "Total DSM (MWh saved)": float(disp_df[dsm_cols].sum().sum()) if dsm_cols else 0.0,
            "Total Storage Charging (MWh)": float(disp_df[storage_chg_cols].sum().sum()) if storage_chg_cols else 0.0,
            "Total Storage Discharging (MWh)": float(disp_df[storage_dis_cols].sum().sum()) if storage_dis_cols else 0.0,
            "Total EV Charging (MWh)": float(disp_df[ev_col].sum()) if ev_col else 0.0,
            "Total Imported Hydro (MWh)": float(disp_df["Imported Hydro"].sum()) if "Imported Hydro" in disp_df.columns else 0.0,
            "Max Imported Hydro (MW)": float(disp_df["Imported Hydro"].max()) if "Imported Hydro" in disp_df.columns else 0.0,
        }

        print("\n=== DISPATCH TOTALS (Daily MWh) ===")
        print(pd.DataFrame({"Metric": totals.keys(), "MWh": totals.values()}).to_string(index=False))
        print("==================================\n")

    return inv_df, disp_df

