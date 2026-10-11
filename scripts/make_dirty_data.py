import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "sample_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

n_sales = 1200
start_date = datetime(2025, 1, 1)

dates = [start_date + timedelta(days=int(i * 365 / n_sales)) for i in range(n_sales)]
date_strs = [d.strftime("%Y-%m-%d") if random.random() > 0.5 else d.strftime("%d/%m/%Y") for d in dates]

cities = np.random.choice(
    ["Baku", "BAKU", "Ganja", "Sumqayit", "Baku ", " Baku", np.nan],
    size=n_sales,
    p=[0.35, 0.25, 0.1, 0.1, 0.05, 0.05, 0.1],
)

prices = np.random.exponential(scale=50, size=n_sales) + 10
outlier_indices = np.random.choice(n_sales, size=15, replace=False)
prices[outlier_indices] = prices[outlier_indices] * 120

quantities = np.random.choice(
    [1, 2, 3, 4, 5, np.nan, "N/A"],
    size=n_sales,
    p=[0.45, 0.2, 0.1, 0.05, 0.05, 0.07, 0.08],
)

notes = np.array([np.nan if random.random() > 0.1 else f"Qeyd {i}" for i in range(n_sales)])

df_sales = pd.DataFrame(
    {
        "TransactionID": range(1000, 1000 + n_sales),
        "Date": date_strs,
        "City": cities,
        "Price": prices,
        "Quantity": quantities,
        "Notes": notes,
    }
)

temp_dates = pd.to_datetime(df_sales["Date"], errors="coerce", format="mixed")
max_date = temp_dates.max()
last_month_mask = (temp_dates.dt.year == max_date.year) & (temp_dates.dt.month == max_date.month)
df_sales.loc[last_month_mask, "Price"] *= 1.12

df_sales = pd.concat([df_sales, df_sales.sample(n=30, random_state=42)], ignore_index=True)
df_sales.to_csv(OUT_DIR / "dirty_sales.csv", index=False)

n_cust = 300
cust_ids = range(1, n_cust + 1)
names = [f"  Müştəri {i} " if i % 3 == 0 else f"Müştəri {i}" for i in range(n_cust)]
ages = np.random.choice(
    [22, 25, 30, 35, 42, 250, -3, "unknown", np.nan],
    size=n_cust,
    p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.03, 0.03, 0.07, 0.07],
)
genders = np.random.choice(
    ["M", "male", "MALE", "F", "female", "FEMALE", np.nan],
    size=n_cust,
    p=[0.2, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1],
)

df_cust = pd.DataFrame({"CustomerID": cust_ids, "Name": names, "Age": ages, "Gender": genders})
df_cust.to_excel(OUT_DIR / "dirty_customers.xlsx", index=False)

df_hr = pd.DataFrame(
    {
        "EmpID": [101, 102, 103, 104, 105],
        "Name": [" Anar ", "Aysel", "Vüsal ", "Nigar", "Orxan"],
        "Department": ["IT", "HR", "Sales", "IT", "Finance"],
        "Salary": [1200, 950, np.nan, 1500, 1100],
        "JoinDate": ["2023-01-15", "15/05/2022", "2024-03-10", "yanlis_tarix", "2023-11-01"],
    }
)
df_hr.to_csv(OUT_DIR / "dirty_hr.csv", index=False)

print("Files created in:", OUT_DIR)
for name in ["dirty_sales.csv", "dirty_customers.xlsx", "dirty_hr.csv"]:
    print(" -", name, (OUT_DIR / name).stat().st_size, "bytes")