# 📈 Utility Investment Optimization & Dispatch Model — Shiny for Python

Multi-season optimization model for evaluating utility-scale investments in solar, wind, net metering, battery storage, and pumped hydro — with an integrated daily dispatch LP.

This project combines capacity planning and hourly dispatch modelling into a single analytical tool, designed for utility planners and energy consultants evaluating investment pathways under cost, GHG, and reliability objectives.

🚀 Live Demo (GIF to be added)

📌 A 5–12 second GIF should go here showing:

Selecting objective type

Running the optimization

Viewing investment table + supply stack chart

Navigating to the dispatch tab

(We will generate this once the dashboard is live.)

## 🌍 Purpose & Use Cases

Traditional long-term planning tools often evaluate technologies one by one or rely on static assumptions.

This model solves two linked problems:

#### 1️⃣ Investment LP:
Optimizes installed capacities of technologies under cost or GHG objectives.

#### 2️⃣ Dispatch LP:
Checks hourly feasibility across a 24-hour representative day for each season (winter, summer, fall, spring), ensuring that load is met and storage behaves correctly.

#### Real-world applications

Utility planners can use this tool to:
- Build GHG-minimizing portfolios aligned with Net Zero objectives
- Estimate optimal mix of solar / wind / net metering / storage
- Evaluate cost trade-offs using LCOE-based or variable-cost objectives
- Test sensitivity to discount rate and load growth scenarios
- Visualize seasonal reliability constraints using dispatch plots
- Produce GHG abatement cost curves for decision makers

## 🧩 High-Level Architecture
+-------------------------------------------------------------+
|                   Investment Optimization LP                |
|-------------------------------------------------------------|
| Objective: Minimize Cost / Minimize GHG / Weighted Combo    |
|                                                             |
| Decision Variables:                                         |
|   - Installed Capacity (MW) for each technology             |
|                                                             |
| Constraints:                                                |
|   - Seasonal energy availability                            |
|   - Capital cost budget (optional)                          |
|   - Capacity factors (proxy or dispatch-based)              |
+---------------------------+---------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|                   Dispatch Optimization LP                  |
|-------------------------------------------------------------|
| Hourly simulation for selected season:                      |
|  - Load balance (supply + discharge = load + charge)        |
|  - Storage charging/discharging limits                      |
|  - SOC continuity (SOC_end = SOC_start)                     |
|  - Availability limits for solar, wind, net metering        |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|                      Dashboard & Outputs                    |
|-------------------------------------------------------------|
|  - Investment summary table                                 |
|  - Supply stack chart                                       |
|  - GHG abatement curve                                      |
|  - Seasonal dispatch plots                                  |
|  - Scenario saving (JSON)                                   |
+-------------------------------------------------------------+

## 🔧 Optimization Model

### Decision Variables

##### Investment LP
- CapTech[t] — Installed capacity (MW) for technology t
(Solar_PV, Wind, Net_Metering, Battery_Storage, Pumped_Hydro)

#### Dispatch LP
- Dispatch[t, h] — Hourly output (MW)
- Charge[t, h], Discharge[t, h] — For storage technologies
- SOC[t, h] — State of charge

### 🎯 Objective Functions

Users select one of:

#### 1. Cost Minimization

Uses capital + variable cost streams:
```
Minimize Σ(t) [CapitalCost[t] * CapTech[t] + 
               Σ(h) VariableCost[t] * Dispatch[t,h]]
```
#### 2. GHG Minimization

Minimizes tonnes of CO₂ displaced or emitted:
```
Minimize Σ(t) [GHGIntensity[t] * EnergyProduced[t]]
```
#### 3. Weighted Objective

A convex combination:
```
Obj = α * Cost + (1 - α) * GHG
```

Where α is chosen with a slider in the UI.

### 📏 Core Constraints

#### Load Balance (per hour)
```
Σ_t Dispatch[t,h] + Discharge[h] = Load[h] + Charge[h]
```
#### Storage Constraints
- Charge/discharge limits
- Round-trip efficiency
- SOC bounds
- End-of-day SOC = start-of-day SOC (seasonal balance)

#### Availability Limits
```
Dispatch[t,h] ≤ CapTech[t] * Availability[t,h]
```

#### Non-negativity and Capacity Bounds
```
CapTech[t] ≥ 0
Dispatch[t,h] ≥ 0
```

## 📊 Dashboard Features

### Investment Summary Table

Shows optimized capacities, annual energy output, costs, and GHG effects.

### Supply Stack Chart

A clear, stacked bar visualization of optimized generation mix.

### GHG Abatement Curve

Plots incremental abatement vs incremental cost.

### Seasonal Dispatch Plot

For each season (winter/summer/fall/spring):
- Solar + wind + net metering
- Battery + pumped hydro charge/discharge (negative = charging)
- Load curve overlay
- Visual inspection of hourly reliability

### Scenario Saving

Outputs input assumptions + LP results to a JSON file.

## 📸 Example Outputs (placeholders)

Replace with images under /assets/:

![Investment Summary](assets/invest_summary.png)
![Supply Stack](assets/supply_stack.png)
![GHG Abatement Curve](assets/ghg_abatement.png)
![Winter Dispatch](assets/dispatch_winter.png)
```
🗂️ Project Structure
investment_dashboard/
├── app/
│   └── app.py                      # Shiny UI + server
├── solver/
│   ├── solve_investment_lp.py      # Investment LP
│   ├── solve_dispatch_lp.py        # Dispatch LP
│   └── utils.py                    # Shared functions
├── data/
│   ├── tech_parameters_split_costs.csv
│   ├── load_curve_winter.csv
│   ├── availability_winter.csv
│   └── ... (other seasonal files)
├── assets/                         # GIFs and screenshots
├── requirements.txt
├── README.md                       # (this file)
└── .gitignore
```

### ⚙️ Installation & Running Locally

#### 1. Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
#### 2. Install dependencies
```
pip install -r requirements.txt
```
#### 3. Run the dashboard
```
python3 -m shiny run --reload app/app.py
```
#### Then visit:

👉 http://127.0.0.1:8000

## 🧠 Modeling Notes

- Fully reproducible LP implemented using PuLP
- Supports seasonal load curves provided as CSV
- Availability profiles imported from seasonal datasets
- Dispatch feasibility ensures realistic capacity factors
- Backend functions designed for notebook-based scenario studies

## 📄 License

MIT License

## 👤 Author

Tony Peluso, PhD
Energy Modelling & Grid Analytics — Montreal, QC
📧 tonympeluso@gmail.com

🔗 GitHub: https://github.com/TonyMPeluso

🔗 LinkedIn: https://www.linkedin.com/in/tony-peluso-phd
