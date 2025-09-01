# Hedge_Fund_Optimization
Portfolio optimization pipelines for UCITS &amp; Offshore USD funds. Features: corrected Sortino &amp; Sharpe, NaN-aware reweighting, auto stress/relative stress shading, strategy constraints, regression trend tests, and benchmark comparison.

# Portfolio Optimizer — UCITS & Offshore

🚀 Python pipelines for hedge fund portfolio optimization under **UCITS** and **Offshore** strategy rules.  

This repository contains the **raw code only**.  
👉 You will need to supply your own **returns dataset** (Excel file) and optional **benchmark file** in order to run the scripts.  

---

## ✨ Features

- **Sharpe & Sortino Optimization**
  - Corrected Sortino (downside semideviation, no centering).
  - Annualized Sharpe with risk-free adjustment.

- **Strategy Allocation Constraints**
  - UCITS & Offshore strategy allocation bounds.
  - Per-fund weights (default: 5–25%).
  - Fund count limits (default: 4–20).

- **Robust Data Handling**
  - NaN-aware monthly reweighting.
  - Auto-detects percent vs decimal returns.
  - Minimum observation cutoff for fund eligibility.

- **Analytics & Stress Testing**
  - Auto-calibrated stress periods (benchmark drawdowns, rolling returns, vol spikes).
  - Relative stress detection (portfolio underperformance).
  - Regression tests: “Do fewer funds = better performance?”
  - k-comparison plots vs benchmark.

- **Visualization**
  - Cumulative growth vs benchmark (with stress shading).
  - Rolling correlation vs benchmark.
  - Distributions for ratios, returns, volatilities, and fund counts.

---

## 📂 Repository Structure
Hedge_Fund_Optimization/

│ ├─ ucits.py # UCITS USD optimizer

│ └─ offshore.py # Offshore USD optimizer

├─ requirements.txt # Dependencies

└─ README.md


---

## ⚡ Usage
1. Clone the repo:
```bash
git clone https://github.com/andreasaxod/Hedge_Fund_Optimization
cd Hedge_Fund_Optimization

2. Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run either script
python ucits.py
python offshore.py


