# Investment Dashboard

This **Shiny for Python** dashboard supports investment planning in electricity generation technologies. It includes tools for optimizing an investment portfolio based on **cost**, **GHG abatement**, or a **weighted combination** of both. The dashboard also features an **hourly dispatch model** for visualizing load coverage using selected technologies.

---

## 🚀 Features

- Choose objective: **Cost Minimization**, **GHG Abatement**, or a **Weighted** objective.
- Set **discount rate** and **demand growth scenario** (Low, Medium, High).
- Visualize:
  - Summary investment results
  - Detailed technology contributions
  - Power supply stack chart
  - GHG abatement curve
- (Coming soon) View **hour-by-hour dispatch results** based on load curves.
- Save and compare **multiple investment scenarios**.

---

## 📦 Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/TonyMPeluso/investment_dashboard.git
cd investment_dashboard
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt