# 📦 Inventory Management · DQN Agent

A desktop application for intelligent inventory management using a **Deep Q-Network (DQN)** reinforcement learning agent. The agent learns optimal order quantities from historical demand data and provides daily recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)

---

## ✨ Features

- **DQN agent** with experience replay, target network, and ε-greedy exploration
- **Seasonal demand modeling** — sine/cosine encoding of day-of-year
- **7-day demand forecast** based on historical seasonal averages
- **Interactive daily dashboard** — confirm orders, input actual sales, track KPIs
- **Live training charts** — reward curve, epsilon decay
- **Simulation analytics** — inventory vs demand, daily profit, cumulative shortage

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/inventory-dqn.git
cd inventory-dqn
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python inventory_app.py
```

---

## 📊 Data Format

The application reads an **Excel file** (`.xlsx`) with a sheet named **`State_Input`**.

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date of the record |
| `yesterday_demand` | int/float | Demand from the previous day |
| `day_of_year` | int | Day number in the year (1–365) |

A sample file `data.xlsx` is included in the repository.

---

## ⚙️ Configurable Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Warehouse capacity | Maximum inventory units | 3000 |
| Initial inventory | Starting stock level | 500 |
| Lead time (days) | Delivery delay | 3 |
| Selling price | Revenue per unit sold | 100 |
| Holding cost | Cost per unit stored per day | 2 |
| Order unit cost | Cost per unit ordered | 60 |
| Order fixed cost | Fixed cost per order placed | 200 |
| Stockout penalty | Penalty per unit of unmet demand | auto |
| Episodes | Number of training episodes | 300 |

---

## 🧠 Model Architecture

```
State (19 features):
  - Normalized inventory level
  - Seasonal encoding (sin/cos of day-of-year)
  - Rolling demand ratios (7, 14, 30 days)
  - Last demand ratio
  - Short-term demand trend
  - 7-day seasonal demand forecast
  - Pending orders (quantity & count)

DQN:
  Linear(19 → 64) → ReLU
  Linear(64 → 64)  → ReLU
  Linear(64 → N_actions)

Actions: [0, 50, 100, ..., capacity/3] units
```

Training uses **Double DQN** with:
- Replay buffer: 50 000 transitions
- Batch size: 64
- Target network update: every 10 episodes
- Optimizer: Adam (lr = 5e-4)
- Gradient clipping: norm ≤ 1.0

---

## 📁 Project Structure

```
inventory-dqn/
├── inventory_app.py   # Main application (UI + DQN + environment)
├── data.xlsx          # Sample demand dataset
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Requirements

- Python 3.9+
- PyQt5
- PyTorch
- NumPy
- Pandas
- Matplotlib

See `requirements.txt` for pinned versions.

